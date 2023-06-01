import torch
import torch.nn as nn


class SI_SNR(nn.Module):
    def __init__(self):
        super(SI_SNR, self).__init__()
        self.EPS = 1e-8

    def forward(self, source, estimate_source):
        if source.shape[-1] > estimate_source.shape[-1]:
            source = source[..., :estimate_source.shape[-1]]
        if source.shape[-1] < estimate_source.shape[-1]:
            estimate_source = estimate_source[..., :source.shape[-1]]

        # step 1: Zero-mean norm
        source = source - torch.mean(source, dim=-1, keepdim=True)
        estimate_source = estimate_source - torch.mean(estimate_source, dim=-1, keepdim=True)

        # step 2: Cal si_snr
        # s_target = <s', s>s / ||s||^2
        ref_energy = torch.sum(source ** 2, dim = -1, keepdim=True) + self.EPS
        proj = torch.sum(source * estimate_source, dim = -1, keepdim=True) * source / ref_energy
        # e_noise = s' - s_target
        noise = estimate_source - proj
        # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
        ratio = torch.sum(proj ** 2, dim = -1) / (torch.sum(noise ** 2, dim = -1) + self.EPS)
        sisnr = 10 * torch.log10(ratio + self.EPS)

        return 0 - torch.mean(sisnr)

class MuSE_loss(nn.Module):
    def __init__(self):
        super(MuSE_loss, self).__init__()
        self.si_snr_loss = SI_SNR()
        self.speaker_loss = nn.CrossEntropyLoss()

    def forward(self, tgt_wav, pred_wav, tgt_spk, pred_spk):
        si_snr = self.si_snr_loss(tgt_wav, pred_wav)
        ce = self.speaker_loss(pred_spk[0], tgt_spk) + self.speaker_loss(pred_spk[1], tgt_spk) + self.speaker_loss(pred_spk[2], tgt_spk) + self.speaker_loss(pred_spk[3], tgt_spk)
        return {'si_snr': si_snr, 'ce': ce} 

