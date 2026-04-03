import random, logging, numpy as np, torch
from torch.utils.data import DataLoader
from src.generative.config import GenerativeExperimentConfig
from src.generative.dataset import GenerativeDataset, generative_collate
from src.generative.models.generative_model import GenerativeModel
from src.generative.training.trainer import GenerativeTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

config = GenerativeExperimentConfig.from_yaml('configs/generative/exp5_full_context.yaml')
seed = config.training.seed
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

ds_kwargs = dict(use_full_context=config.model.use_full_context,
    use_simplified_context=config.model.use_simplified_context,
    use_scoring_events_only=config.model.use_scoring_events_only,
    max_scoring_events=config.model.max_scoring_events)
train_ds = GenerativeDataset(config.data, split='train', **ds_kwargs)
val_ds = GenerativeDataset(config.data, split='val', **ds_kwargs)
test_ds = GenerativeDataset(config.data, split='test', **ds_kwargs)

loader_kwargs = dict(batch_size=config.data.batch_size, num_workers=config.data.num_workers,
    pin_memory=config.data.pin_memory, collate_fn=generative_collate)
train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs)
val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

model = GenerativeModel(config.model)
trainer = GenerativeTrainer(model, config, train_loader, val_loader, test_loader)

# Resume from best checkpoint
resume_path = 'checkpoints/generative/gen_exp5b_clock_delta/best.pt'
logger.info(f'Resuming from {resume_path}')
trainer.load_checkpoint(resume_path)

final_metrics = trainer.train()
logger.info('=' * 60)
for k, v in sorted(final_metrics.items()):
    logger.info(f'  {k}: {v:.4f}')
logger.info('=' * 60)
