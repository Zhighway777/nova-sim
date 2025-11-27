from bossa_nova.config import BossaNovaConfig
from bossa_nova.data_visual.post_processor import PostProcessor
from bossa_nova.executor.bossa_nova_executor import BossaNovaExecutor
from bossa_nova.utils.config_utils import load_config
from tests.test_base import CaseInfo
import logging
logger = logging.getLogger(__name__)


def test_parallel_cache(outdir, force_rerun, rank_id, device):
    case_info = CaseInfo("linear", [1,  4096, 12288, 1536], None, None, None, [
                         "ci case", "libra_zebu"], enable_cache=True, outdir=outdir)
    case_info.do_sim(force_rerun, rank_id, device)
