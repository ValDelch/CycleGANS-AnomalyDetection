from metrics import InceptionV3, calculate_fretchet
import torch

def infer(model_name, model, normal_dataloader, abnormal_dataloader, device):

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048] #64 192 768 2048
    model_InceptionV3 = InceptionV3([block_idx])
    model_InceptionV3 = model_InceptionV3.to(device)

    OK, NG = {'sse': [], 'fid': []}, {'sse': [], 'fid': []}

    if model_name == 'cgan256' or model_name == 'cgan64':
        get_maps = get_maps_cgan
    elif model_name == 'ganomaly':
        get_maps = get_maps_ganomaly
    elif model_name == 'patchcore':
        get_maps = get_maps_patchcore
    elif model_name == 'padim':
        get_maps = get_maps_padim
    else:
        raise ValueError('Unknown model name: {}'.format(model_name))

    model.eval()
    maps_normal = {}
    for data in normal_dataloader:
        data = data['image'].to(device)
        _maps = get_maps(model, data, model_InceptionV3, device)
        if maps_normal == {}:
            maps_normal = _maps.copy()
        else:
            for key in maps_normal.keys():
                maps_normal[key] += _maps[key]

    maps_abnormal = {}
    for data in abnormal_dataloader:
        data = data['image'].to(device)
        _maps = get_maps(model, data, model_InceptionV3, device)
        if maps_abnormal == {}:
            maps_abnormal = _maps.copy()
        else:
            for key in maps_abnormal.keys():
                maps_abnormal[key] += _maps[key]

    return maps_normal, maps_abnormal


def get_maps_cgan(model, data, inception, device):

    if device == 'cuda':
        use_cuda = True
    else:
        use_cuda = False

    fake = model(data)
    sse = []
    fid = []
    for j in range(data.shape[0]):
        _sse = (data[j][None,:,:,:].detach() - fake[j][None,:,:,:].detach())**2
        sse.append(round(torch.sum(_sse).item(),2))
        _fid = calculate_fretchet(data[j][None,:,:,:], 
                                  fake[j][None,:,:,:], 
                                  inception, 
                                  cuda=use_cuda)
        fid.append(round(torch.sum(_fid).item(),2))

    return {'sse': sse, 'fid': fid}


def get_maps_ganomaly(model, data, inception, device):

    if device == 'cuda':
        use_cuda = True
    else:
        use_cuda = False

    real, fake, _, _ = model(data)
    sse = []
    fid = []
    for j in range(data.shape[0]):
        _sse = (real[j][None,:,:,:].detach() - fake[j][None,:,:,:].detach())**2
        sse.append(round(torch.sum(_sse).item(),2))
        _fid = calculate_fretchet(real[j][None,:,:,:], 
                                  fake[j][None,:,:,:], 
                                  inception, 
                                  cuda=use_cuda)
        fid.append(round(torch.sum(_fid).item(),2))

    return {'sse': sse, 'fid': fid}


def get_maps_patchcore(model, data, inception, device):

    _sse, _ = model(data)
    sse = []
    for _ in range(data.shape[0]):
        sse.append(round(torch.sum(_sse).item(),2))

    return {'sse': sse}


def get_maps_padim(model, data, inception, device):
    
    _sse = model(data)
    sse = []
    for _ in range(data.shape[0]):
        sse.append(round(torch.sum(_sse).item(),2))

    return {'sse': sse}