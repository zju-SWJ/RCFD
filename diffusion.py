import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


def make_beta_cosine(T, cosine_s=8e-3):
    timesteps = (torch.arange(T + 1, dtype=torch.float64) / T + cosine_s)
    alphas = timesteps / (1 + cosine_s) * math.pi / 2
    alphas = torch.cos(alphas).pow(2)
    alphas = alphas / alphas[0]
    betas = 1 - alphas[1:] / alphas[:-1]
    betas = betas.clamp(max=0.999)
    return betas
    

class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, T, time_scale, loss_type, mean_type):
        super().__init__()

        self.model = model
        self.T = T
        self.time_scale = time_scale
        self.loss_type = loss_type
        self.mean_type = mean_type

        # self.register_buffer(
        #    'betas', torch.linspace(beta_1, beta_T, T).double())
        self.register_buffer(
            'betas', make_beta_cosine(self.T * self.time_scale + 1)) # change from T to T+1
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))
    
    def predict_eps_from_x(self, x_t, x_0, t):
        # x_t = sqrt_alphas_bar * x_0 + sqrt_one_minus_alphas_bar * eps
        # -> eps = (x_t - sqrt_alphas_bar * x_0) / sqrt_one_minus_alphas_bar
        eps = (
            (x_t - extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0) / 
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape))
        return eps

    def forward(self, x_0, y=-1): # predict x_0 directly
        t = torch.randint(self.T + 1, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = (
            extract(self.sqrt_alphas_bar, t * self.time_scale, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t * self.time_scale, x_0.shape) * noise)
        if self.mean_type == 'xstart':
            x_0_rec = self.model(x_t, t * self.time_scale, y) # x_hat in PD page 5
            # x_0_rec = torch.clip(x_0_rec, -1, 1) # clip should not be used during training?
            bs = x_0.size(0)
            loss_x_0 = torch.mean(F.mse_loss(x_0_rec, x_0, reduction='none').reshape(bs, -1), dim=-1)
            loss_eps = torch.mean(F.mse_loss(self.predict_eps_from_x(x_t, x_0_rec, t * self.time_scale), noise, reduction='none').reshape(bs, -1), dim=-1)
            if self.loss_type == 'x':
                return torch.mean(loss_x_0)
            elif self.loss_type == 'eps':
                return torch.mean(loss_eps)
            elif self.loss_type == 'both':
                return torch.mean(torch.maximum(loss_x_0, loss_eps)) # truncated SNR weighting
        elif self.mean_type == 'epsilon':
            eps = self.model(x_t, t * self.time_scale, y)
            return F.mse_loss(eps, noise)


class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, T, time_scale, img_size=32,
                 mean_type='xstart', var_type='fixedlarge', loss_type='both'):
        assert mean_type in ['xprev', 'xstart', 'epsilon']
        assert var_type in ['fixedlarge', 'fixedsmall']
        super().__init__()

        self.model = model
        self.T = T
        self.img_size = img_size
        self.mean_type = mean_type
        self.var_type = var_type
        self.time_scale = time_scale
        self.loss_type = loss_type

        # self.register_buffer(
        #     'betas', torch.linspace(beta_1, beta_T, T).double())
        self.register_buffer(
            'betas', make_beta_cosine(T * time_scale + 1)) # change from T to T+1
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_recip_alphas_bar', torch.sqrt(1. / alphas_bar))
        self.register_buffer(
            'sqrt_recipm1_alphas_bar', torch.sqrt(1. / alphas_bar - 1))


        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T * time_scale + 1]

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer(
            'posterior_var',
            self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer(
            'posterior_log_var_clipped',
            torch.log(
                torch.cat([self.posterior_var[1:2], self.posterior_var[1:]])))
                ## self.posterior_var))
        self.register_buffer(
            'posterior_mean_coef1',
            torch.sqrt(alphas_bar_prev) * self.betas / (1. - alphas_bar))
        self.register_buffer(
            'posterior_mean_coef2',
            torch.sqrt(alphas) * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def q_mean_variance(self, x_0, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior
        q(x_{t-1} | x_t, x_0)
        """
        assert x_0.shape == x_t.shape
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_log_var_clipped = extract(
            self.posterior_log_var_clipped, t, x_t.shape)
        return posterior_mean, posterior_log_var_clipped

    def predict_xstart_from_eps(self, x_t, eps, t):
        assert x_t.shape == eps.shape
        return (
            extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * eps
        )
    
    def predict_eps_from_x(self, x_t, x_0, t):
        # x_t = sqrt_alphas_bar * x_0 + sqrt_one_minus_alphas_bar * eps
        # -> eps = (x_t - sqrt_alphas_bar * x_0) / sqrt_one_minus_alphas_bar
        eps = (
            (x_t - extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0) / 
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape))
        return eps

    def predict_xstart_from_xprev(self, x_t, xprev, t):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            extract(
                1. / self.posterior_mean_coef1, t, x_t.shape) * xprev -
            extract(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t,
                x_t.shape) * x_t
        )

    def p_mean_variance(self, x_t, t, clip, y):
        # below: only log_variance is used in the KL computations
        model_log_var = {
            # for fixedlarge, we set the initial (log-)variance like so to get a better decoder log likelihood
            'fixedlarge': torch.log(torch.cat([self.posterior_var[1:2], self.betas[1:]])),
            ## 'fixedlarge': torch.log(torch.cat([self.posterior_var[0:1], self.betas[1:]])),
            'fixedsmall': self.posterior_log_var_clipped,
        }[self.var_type]
        model_log_var = extract(model_log_var, t * self.time_scale, x_t.shape)

        # Mean parameterization
        if self.mean_type == 'xprev':       # the model predicts x_{t-1}
            x_prev = self.model(x_t, t * self.time_scale, y)
            x_0 = self.predict_xstart_from_xprev(x_t, x_prev, t * self.time_scale)
            model_mean = x_prev
        elif self.mean_type == 'xstart':    # the model predicts x_0
            x_0 = self.model(x_t, t * self.time_scale, y)
            if clip:
                x_0 = torch.clip(x_0, -1., 1.)
            model_mean, _ = self.q_mean_variance(x_0, x_t, t * self.time_scale)
        elif self.mean_type == 'epsilon':   # the model predicts epsilon
            eps = self.model(x_t, t * self.time_scale, y)
            x_0 = self.predict_xstart_from_eps(x_t, eps, t * self.time_scale)
            if clip:
                x_0 = torch.clip(x_0, -1., 1.)
            model_mean, _ = self.q_mean_variance(x_0, x_t, t * self.time_scale)
        else:
            raise NotImplementedError(self.mean_type)

        return model_mean, model_log_var
        
    def forward(self, z_t, t, y):
        return self.model(z_t, t, y)
        
    def ddpm(self, x_T, clip=True, y=-1): # DDPM sampler with stride 1, NOT used in our paper, NOT fully tested
        x_t = x_T
        for time_step in reversed(range(0, self.T + 1)): 
            # it actually uses self.T+1 steps
            # if uses range(1, self.T + 1) or range(0, self.T), sampling may go wrong when self.T is small
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, log_var = self.p_mean_variance(x_t=x_t, t=t, clip=clip, y=y)
            # no noise when t == 0
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.exp(0.5 * log_var) * noise
        x_0 = x_t
        return torch.clip(x_0, -1, 1)

    def ddim(self, x_T, stride, clip=True, y=-1): # DDIM sampler with large stride
        x_t = x_T
        for time_step in reversed(range(stride, self.T + 1, stride)):
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            s = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * (time_step - stride)
            if self.mean_type == 'xstart':
                x_0 = self.model(x_t, t * self.time_scale, y)
                if clip:
                    x_0 = torch.clip(x_0, -1., 1.)
                eps = self.predict_eps_from_x(x_t, x_0, t * self.time_scale)
            elif self.mean_type == 'epsilon':
                eps = self.model(x_t, t * self.time_scale, y)
                x_0 = self.predict_xstart_from_eps(x_t, eps, t * self.time_scale)
                if clip:
                    x_0 = torch.clip(x_0, -1., 1.)
                    eps = self.predict_eps_from_x(x_t, x_0, t * self.time_scale)
            x_t = (
                extract(self.sqrt_alphas_bar, s * self.time_scale, x_0.shape) * x_0 +
                extract(self.sqrt_one_minus_alphas_bar, s * self.time_scale, x_0.shape) * eps)
        x_0 = x_t
        return torch.clip(x_0, -1, 1)
    
    def ddim_each_step(self, x_T, stride, clip=True, y=-1): # DDIM sampler with large stride, record each step
        x_t = x_T
        imgs = []
        for time_step in reversed(range(stride, self.T + 1, stride)):
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            s = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * (time_step - stride)
            if self.mean_type == 'xstart':
                x_0 = self.model(x_t, t * self.time_scale, y)
                if clip:
                    x_0 = torch.clip(x_0, -1., 1.)
                eps = self.predict_eps_from_x(x_t, x_0, t * self.time_scale)
            elif self.mean_type == 'epsilon':
                eps = self.model(x_t, t * self.time_scale, y)
                x_0 = self.predict_xstart_from_eps(x_t, eps, t * self.time_scale)
                if clip:
                    x_0 = torch.clip(x_0, -1., 1.)
                    eps = self.predict_eps_from_x(x_t, x_0, t * self.time_scale)
            # imgs.append(torch.clip(x_0, -1, 1))
            x_t = (
                extract(self.sqrt_alphas_bar, s * self.time_scale, x_0.shape) * x_0 +
                extract(self.sqrt_one_minus_alphas_bar, s * self.time_scale, x_0.shape) * eps)
            imgs.append(torch.clip(x_t, -1, 1))
        x_0 = x_t
        return imgs
    
    def distill(self, student, x_0, y=-1):
        t = 2 * torch.randint(1, student.module.T + 1, (x_0.shape[0],), device=x_0.device)
        # take teacher.T=512, student.T=256 for example, t \in 2 * [1, 2, ..., 256] = [2, 4, ..., 512]
        noise = torch.randn_like(x_0)
        with torch.no_grad():
            z_t = ( # start noise
                extract(self.sqrt_alphas_bar, t * self.time_scale, x_0.shape) * x_0 +
                extract(self.sqrt_one_minus_alphas_bar, t * self.time_scale, x_0.shape) * noise) # create noise pictures, z_t in PD page 4
            if self.mean_type == 'xstart':
                x_0_rec = self.model(z_t, t * self.time_scale, y) # predicted x_0
                # x_0_rec = torch.clip(x_0_rec, -1, 1)
                eps_rec = self.predict_eps_from_x(z_t, x_0_rec, t * self.time_scale) # predicted eps
            elif self.mean_type == 'epsilon':
                eps_rec = self.model(z_t, t * self.time_scale, y)
                x_0_rec = self.predict_xstart_from_eps(z_t, eps_rec, t * self.time_scale)
                # x_0_rec = torch.clip(x_0_rec, -1, 1)
            z_t_minus_1 = (
                extract(self.sqrt_alphas_bar, (t - 1) * self.time_scale, x_0.shape) * x_0_rec +
                extract(self.sqrt_one_minus_alphas_bar, (t - 1) * self.time_scale, x_0.shape) * eps_rec) # get z_t' in PD page 4
            
            if self.mean_type == 'xstart':
                x_0_rec_rec = self.model(z_t_minus_1, (t - 1) * self.time_scale, y)
                # x_0_rec_rec = torch.clip(x_0_rec_rec, -1, 1)
                eps_rec_rec = self.predict_eps_from_x(z_t_minus_1, x_0_rec_rec, (t - 1) * self.time_scale)
            elif self.mean_type == 'epsilon':
                eps_rec_rec = self.model(z_t_minus_1, (t - 1) * self.time_scale, y)
                x_0_rec_rec = self.predict_xstart_from_eps(z_t_minus_1, eps_rec_rec, (t - 1) * self.time_scale)
                # x_0_rec_rec = torch.clip(x_0_rec_rec, -1, 1)
            z_t_minus_2 = (
                extract(self.sqrt_alphas_bar, (t - 2) * self.time_scale, x_0.shape) * x_0_rec_rec +
                extract(self.sqrt_one_minus_alphas_bar, (t - 2) * self.time_scale, x_0.shape) * eps_rec_rec) # get z_t'' in PD page 4

            frac = extract(self.sqrt_one_minus_alphas_bar, (t - 2) * self.time_scale, x_0.shape) / extract(self.sqrt_one_minus_alphas_bar, t * self.time_scale, x_0.shape)
            x_target = (z_t_minus_2 - frac * z_t) / (extract(self.sqrt_alphas_bar, (t - 2) * self.time_scale, x_0.shape) - frac * extract(self.sqrt_alphas_bar, t * self.time_scale, x_0.shape))
            eps_target = self.predict_eps_from_x(z_t, x_target, t * self.time_scale)
        
        if self.mean_type == 'xstart':
            x_0_predicted = student(z_t, t * self.time_scale, y)
            eps_predicted = self.predict_eps_from_x(z_t, x_0_predicted, t * self.time_scale)
        elif self.mean_type == 'epsilon':
            eps_predicted = student(z_t, t * self.time_scale, y)
            x_0_predicted = self.predict_xstart_from_eps(z_t, eps_predicted, t * self.time_scale)
        bs = x_0.size(0)
        loss_x_0 = torch.mean(F.mse_loss(x_0_predicted, x_target, reduction='none').reshape(bs, -1), dim=-1)
        loss_eps = torch.mean(F.mse_loss(eps_predicted, eps_target, reduction='none').reshape(bs, -1), dim=-1)
        if self.loss_type == 'x':
            return torch.mean(loss_x_0)
        elif self.loss_type == 'eps':
            return torch.mean(loss_eps)
        elif self.loss_type == 'both':
            return torch.mean(torch.maximum(loss_x_0, loss_eps))
    
    def Entropy(self, logit):
        p = F.softmax(logit, dim=1)
        entropy = -(p * torch.log(p)).sum(dim=1)
        return entropy
    
    def Diversity(self, logit):
        p = F.softmax(logit, dim=1)
        p = p.mean(dim=0)
        diversity = (p * torch.log(p)).sum()
        return diversity

    def my_distill_latest(self, student, classifier, x_0, y=-1, temp=0.95, alpha=0, beta=0, feat_div=False, imagenet_cls=False, prediction=False):
        t = 2 * torch.randint(1, student.module.T + 1, (x_0.shape[0],), device=x_0.device)
        # take teacher.T=512, student.T=256 for example, t \in 2 * [1, 2, ..., 256] = [2, 4, ..., 512]
        noise = torch.randn_like(x_0)
        with torch.no_grad():
            z_t = ( # start noise
                extract(self.sqrt_alphas_bar, t * self.time_scale, x_0.shape) * x_0 +
                extract(self.sqrt_one_minus_alphas_bar, t * self.time_scale, x_0.shape) * noise) # create noise pictures, z_t in PD page 4
            if self.mean_type == 'xstart':
                x_0_rec = self.model(z_t, t * self.time_scale, y) # predicted x_0
                # x_0_rec = torch.clip(x_0_rec, -1, 1)
                eps_rec = self.predict_eps_from_x(z_t, x_0_rec, t * self.time_scale) # predicted eps
            elif self.mean_type == 'epsilon':
                eps_rec = self.model(z_t, t * self.time_scale, y)
                x_0_rec = self.predict_xstart_from_eps(z_t, eps_rec, t * self.time_scale)
                # x_0_rec = torch.clip(x_0_rec, -1, 1)
            z_t_minus_1 = (
                extract(self.sqrt_alphas_bar, (t - 1) * self.time_scale, x_0.shape) * x_0_rec +
                extract(self.sqrt_one_minus_alphas_bar, (t - 1) * self.time_scale, x_0.shape) * eps_rec) # get z_t' in PD page 4
            
            if self.mean_type == 'xstart':
                x_0_rec_rec = self.model(z_t_minus_1, (t - 1) * self.time_scale, y)
                # x_0_rec_rec = torch.clip(x_0_rec_rec, -1, 1)
                eps_rec_rec = self.predict_eps_from_x(z_t_minus_1, x_0_rec_rec, (t - 1) * self.time_scale)
            elif self.mean_type == 'epsilon':
                eps_rec_rec = self.model(z_t_minus_1, (t - 1) * self.time_scale, y)
                x_0_rec_rec = self.predict_xstart_from_eps(z_t_minus_1, eps_rec_rec, (t - 1) * self.time_scale)
                # x_0_rec_rec = torch.clip(x_0_rec_rec, -1, 1)
            z_t_minus_2 = (
                extract(self.sqrt_alphas_bar, (t - 2) * self.time_scale, x_0.shape) * x_0_rec_rec +
                extract(self.sqrt_one_minus_alphas_bar, (t - 2) * self.time_scale, x_0.shape) * eps_rec_rec) # get z_t'' in PD page 4

            frac = extract(self.sqrt_one_minus_alphas_bar, (t - 2) * self.time_scale, x_0.shape) / extract(self.sqrt_one_minus_alphas_bar, t * self.time_scale, x_0.shape)
            x_target = (z_t_minus_2 - frac * z_t) / (extract(self.sqrt_alphas_bar, (t - 2) * self.time_scale, x_0.shape) - frac * extract(self.sqrt_alphas_bar, t * self.time_scale, x_0.shape))
            if imagenet_cls:
                x_target = F.interpolate(x_target, size=224, mode='bilinear', align_corners=True)
            p_T, feat_T = classifier(x_target) # teacher prediction
        
        if self.mean_type == 'xstart':
            x_0_predicted = student(z_t, t * self.time_scale, y)
        elif self.mean_type == 'epsilon':
            eps_predicted = student(z_t, t * self.time_scale, y)
            x_0_predicted = self.predict_xstart_from_eps(z_t, eps_predicted, t * self.time_scale)
        if imagenet_cls:
            x_0_predicted = F.interpolate(x_0_predicted, size=224, mode='bilinear', align_corners=True)
        p_S, feat_S = classifier(x_0_predicted) # student prediction


        entropy_loss = self.Entropy(p_S).mean()
        if feat_div:
            div_loss = self.Diversity(feat_S)
        else:
            div_loss = self.Diversity(p_S)
        if prediction:
            feat_loss = F.kl_div(F.log_softmax(p_S, dim=-1), F.softmax(p_T / temp, dim=-1).detach(), reduction='batchmean')
        else:
            feat_loss = F.kl_div(F.log_softmax(feat_S, dim=-1), F.softmax(feat_T / temp, dim=-1).detach(), reduction='batchmean')
        loss = feat_loss + alpha * (beta * entropy_loss + (1 - beta) * div_loss)
        return loss, entropy_loss, div_loss, feat_loss
