#模型工厂，根据配置创建算法实例
def get_model(model_name, args):
    name = model_name.lower()
    if name == "icarl":
        from models.icarl import iCaRL
        return iCaRL(args)
    elif name == "bic":
        from models.bic import BiC
        return BiC(args)
    elif name == "podnet":
        from models.podnet import PODNet
        return PODNet(args)
    elif name == "lwf":
        from models.lwf import LwF
        return LwF(args)
    elif name == "ewc":
        from models.ewc import EWC
        return EWC(args)
    elif name == "wa":
        from models.wa import WA
        return WA(args)
    elif name == "der":
        from models.der import DER
        return DER(args)
    elif name == "finetune":
        from models.finetune import Finetune
        return Finetune(args)
    elif name == "replay":
        from models.replay import Replay
        return Replay(args)
    elif name == "gem":
        from models.gem import GEM
        return GEM(args)
    elif name == "coil":
        from models.coil import COIL
        return COIL(args)
    elif name == "foster":
        from models.foster import FOSTER
        return FOSTER(args)
    elif name == "rmm-icarl":
        from models.rmm import RMM_FOSTER, RMM_iCaRL
        return RMM_iCaRL(args)
    elif name == "rmm-foster":
        from models.rmm import RMM_FOSTER, RMM_iCaRL
        return RMM_FOSTER(args)
    elif name == "fetril":
        from models.fetril import FeTrIL 
        return FeTrIL(args)
    elif name == "pass":
        from models.pa2s import PASS
        return PASS(args)
    elif name == "il2a":
        from models.il2a import IL2A
        return IL2A(args)
    elif name == "ssre":
        from models.ssre import SSRE
        return SSRE(args)
    elif name == "memo":   
        from models.memo import MEMO
        return MEMO(args)
    elif name == "beefiso":
        from models.beef_iso import BEEFISO
        return BEEFISO(args)
    elif name == "simplecil":
        from models.simplecil import SimpleCIL
        return SimpleCIL(args)
    elif name == "acil":
        from models.acil import ACIL
        return ACIL(args)
    elif name == "ds-al":
        from models.dsal import DSAL
        return DSAL(args)
    elif name == "aper_finetune":
        from models.aper_finetune import APER_FINETUNE
        return APER_FINETUNE(args)
    elif name == "tagfex":
        from models.tagfex import TagFex
        return TagFex(args)
    elif name == "adapter_one":
        from models.adapter_one import ADAPTER_ONE
        return ADAPTER_ONE(args)
    elif name == "adapter_one_bic":
        from models.adapter_one_bic import ADAPTER_ONE_BIC
        return ADAPTER_ONE_BIC(args)
    elif name == "adapter_one_loss":
        from models.adapter_one_loss import ADAPTER_ONE_LOSS
        return ADAPTER_ONE_LOSS(args)
    elif name == "adapter_one_lossjubu":
        from models.adapter_one_lossjubu import ADAPTER_ONE_LOSSJUBU
        return ADAPTER_ONE_LOSSJUBU(args)
    elif name == "adapter_one_simplecil":
        from models.adapter_one_simplecil import ADAPTER_ONE_SIMPLECIL
        return ADAPTER_ONE_SIMPLECIL(args)
    # elif name == "adapter_onetest":
    #     from models.adapter_onetest import ADAPTER_ONETEST
    #     return ADAPTER_ONETEST(args)
    elif name == "adapter_one_feature":
        from models.adapter_one_feature import ADAPTER_ONE_FEATURE
        return ADAPTER_ONE_FEATURE(args)
    elif name == "adapter_one_feature_mgda":
        from models.adapter_one_feature_mgda import ADAPTER_ONE_FEATURE_MGDA
        return ADAPTER_ONE_FEATURE_MGDA(args)
    # elif name == "adapter_one_mos":
    #     from models.adapter_one_mos import ADAPTER_ONE_MOS
    #     return ADAPTER_ONE_MOS(args)
    elif name == "adapter_one_feature_measure":
        from models.adapter_one_feature_measure import ADAPTER_ONE_FEATURE_MEASURE
        return ADAPTER_ONE_FEATURE_MEASURE(args)
    elif name == "adapter_one_feature_pstat":
        from models.adapter_one_feature_pstat import ADAPTER_ONE_FEATURE_PSTAT
        return ADAPTER_ONE_FEATURE_PSTAT(args)
    else:
        assert 0
