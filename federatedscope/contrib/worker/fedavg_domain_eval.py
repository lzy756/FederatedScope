"""
FedAvg Server with Domain-Specific Evaluation

This server extends the standard FedAvg server to add domain-specific
evaluation capabilities for Office-Caltech dataset, allowing fair comparison
with cross_domain_adaptive method.
"""

import logging
import numpy as np
from typing import Dict

from federatedscope.core.workers import Server, Client
from federatedscope.register import register_worker

logger = logging.getLogger(__name__)


class FedAvgDomainEvalServer(Server):
    """
    FedAvg Server with domain-specific evaluation for Office-Caltech dataset.

    This server performs standard FedAvg aggregation but adds custom server-side
    evaluation on balanced test sets from each domain (amazon, webcam, dslr, caltech).
    """

    def __init__(self, *args, **kwargs):
        """Initialize server with domain-specific evaluation support."""
        # Handle data argument for server-side evaluation
        # In standalone mode, if server has no data but make_global_eval is True,
        # we need to provide a dummy data dict to pass the assertion in parent class
        data_arg = kwargs.get('data', None)
        cfg_arg = kwargs.get('config', args[1] if len(args) > 1 else None)

        # If make_global_eval is True but data is None, create a placeholder
        if cfg_arg and cfg_arg.federate.make_global_eval and data_arg is None:
            # Create a minimal dummy data dict structure
            # The actual test data will be loaded in _load_balanced_test_data()
            kwargs['data'] = {'test': None}  # Placeholder that will be replaced

        # Call parent constructor
        super().__init__(*args, **kwargs)

        # Load balanced server-side test data for global evaluation
        if self._cfg.federate.make_global_eval and self._cfg.data.type.lower() == 'office_caltech':
            try:
                self._load_balanced_test_data()
            except Exception as e:
                logger.warning(f"Failed to load balanced test data: {e}")
                logger.warning("Server-side evaluation will be disabled or use default behavior")
                # Don't fail initialization, just skip custom server evaluation

    def _load_balanced_test_data(self):
        """
        Load balanced test datasets for server-side global evaluation.

        Creates 4 balanced test sets (one per domain: amazon, webcam, dslr, caltech),
        where each test set has equal number of samples from each class.
        """
        from federatedscope.cv.dataset.office_caltech import load_balanced_office_caltech_data
        from torch.utils.data import DataLoader

        logger.info("Loading balanced test datasets for server-side evaluation...")

        # Get samples per class from config, default to 10
        try:
            samples_per_class = self._cfg.data.server_test_samples_per_class
        except (AttributeError, KeyError):
            samples_per_class = 10
            logger.info(f"Using default samples_per_class={samples_per_class} for server test data")

        # Load balanced test data for all domains
        balanced_datasets = load_balanced_office_caltech_data(
            root=self._cfg.data.root,
            samples_per_class=samples_per_class,
            transform=None,  # Will use default transform
            seed=self._cfg.seed
        )

        # Create data loaders for each domain
        self.test_data_loaders = {}
        for domain, dataset in balanced_datasets.items():
            loader = DataLoader(
                dataset,
                batch_size=self._cfg.data.batch_size,
                shuffle=False,
                num_workers=self._cfg.data.num_workers
            )
            self.test_data_loaders[domain] = loader

        logger.info(f"Loaded balanced test datasets for {len(self.test_data_loaders)} domains: "
                   f"{list(self.test_data_loaders.keys())}")

    def eval(self):
        """
        Override eval to perform server-side evaluation on balanced test sets.

        Evaluates the global model on each domain's balanced test set separately,
        providing domain-specific performance metrics and weighted average.
        """
        # Check if we should perform custom server-side evaluation
        if not (self._cfg.federate.make_global_eval and hasattr(self, 'test_data_loaders')):
            # Fall back to default evaluation - but skip if no data
            logger.info("Skipping server-side evaluation (test_data_loaders not available)")
            return

        # Check if test_data_loaders is empty or None
        if not self.test_data_loaders:
            logger.warning("test_data_loaders is empty, skipping server-side evaluation")
            return

        # Check if trainer exists
        if not hasattr(self, 'trainer') or self.trainer is None:
            logger.warning("Server has no trainer, skipping server-side evaluation")
            return

        try:
            # Perform domain-specific evaluation
            logger.info("="*80)
            logger.info(f"[Round {self.state}] Server-side Evaluation on Balanced Test Sets")
            logger.info("="*80)

            # Store results for each domain
            domain_results = {}
            domain_sizes = {}

            for domain, test_loader in self.test_data_loaders.items():
                # Check if loader has data
                if len(test_loader.dataset) == 0:
                    logger.warning(f"Domain {domain} has empty dataset, skipping")
                    continue

                # Temporarily set the test data for this domain
                original_test_data = self.trainer.ctx.get('test_data')
                original_test_loader = self.trainer.ctx.get('test_loader')
                original_data = self.trainer.ctx.get('data')
                original_cur_split = self.trainer.ctx.get('cur_split')

                # Set test data with proper structure
                # Use ReIterator to wrap the loader for compatibility with trainer
                from federatedscope.core.auxiliaries.ReIterator import ReIterator

                self.trainer.ctx.test_data = test_loader.dataset
                self.trainer.ctx.test_loader = ReIterator(test_loader)
                self.trainer.ctx.cur_split = 'test'

                # CRITICAL: Set num_test_epoch and num_test_batch to ensure evaluation loop runs
                # The trainer's _run_epoch and _run_batch use these to determine loop iterations
                # Must handle None case as well as 0
                if not hasattr(self.trainer.ctx, 'num_test_epoch') or self.trainer.ctx.num_test_epoch in [0, None]:
                    self.trainer.ctx.num_test_epoch = 1

                # Set num_test_batch - this is CRITICAL for the batch loop to run
                if not hasattr(self.trainer.ctx, 'num_test_batch') or self.trainer.ctx.num_test_batch in [0, None]:
                    self.trainer.ctx.num_test_batch = len(test_loader)

                # Also set data dict if it's expected
                if original_data is not None and hasattr(original_data, '__getitem__'):
                    try:
                        original_data['test'] = test_loader.dataset
                    except (TypeError, KeyError):
                        pass

                # Run evaluation
                try:
                    test_metrics = self.trainer.evaluate(target_data_split_name='test')

                    # Add dataset size to metrics for monitor formatting
                    # Monitor expects '{split}_total' key
                    if 'test_total' not in test_metrics:
                        test_metrics['test_total'] = len(test_loader.dataset)

                except Exception as e:
                    logger.error(f"Evaluation failed for domain {domain}: {e}")
                    import traceback
                    traceback.print_exc()
                    # Restore context before continuing
                    self._restore_trainer_context(
                        original_test_data, original_test_loader,
                        original_cur_split, original_data
                    )
                    continue

                # Store domain size for weighted average
                domain_sizes[domain] = len(test_loader.dataset)

                # Extract accuracy (handle different metric formats)
                acc = None
                if 'test_acc' in test_metrics:
                    acc = test_metrics['test_acc']
                elif 'acc' in test_metrics:
                    acc = test_metrics['acc']

                domain_results[domain] = {
                    'accuracy': acc,
                    'metrics': test_metrics,
                    'size': domain_sizes[domain]
                }

                # Format and log results
                from federatedscope.core.auxiliaries.utils import merge_dict_of_results
                formatted_eval_res = self._monitor.format_eval_res(
                    test_metrics,
                    rnd=self.state,
                    role=f'Server #{domain}',
                    forms=self._cfg.eval.report,
                    return_raw=True
                )

                # Update best results with domain tag (only if results are not empty)
                if formatted_eval_res.get('Results_raw'):
                    self._monitor.update_best_result(
                        self.best_results,
                        formatted_eval_res['Results_raw'],
                        results_type=f"server_global_eval_{domain}"
                    )
                else:
                    logger.warning(f"Empty evaluation results for domain {domain}, skipping best result update")

                # Merge with history
                self.history_results = merge_dict_of_results(
                    self.history_results, formatted_eval_res
                )

                # Restore original test data
                self._restore_trainer_context(
                    original_test_data, original_test_loader,
                    original_cur_split, original_data
                )

            # Check if we got any results
            if not domain_results:
                logger.warning("No domain evaluation results obtained, skipping evaluation")
                return

            # Calculate and log weighted average accuracy
            self._log_domain_results(domain_results, domain_sizes)

        except Exception as e:
            logger.error(f"Error during domain-specific evaluation: {e}")
            import traceback
            traceback.print_exc()
            # Skip evaluation on error instead of falling back
            logger.warning("Skipping server-side evaluation due to error")

    def _restore_trainer_context(self, original_test_data, original_test_loader,
                                 original_cur_split, original_data):
        """Restore trainer context to original state."""
        if original_test_data is not None:
            self.trainer.ctx.test_data = original_test_data
        else:
            if 'test_data' in self.trainer.ctx:
                del self.trainer.ctx['test_data']

        if original_test_loader is not None:
            self.trainer.ctx.test_loader = original_test_loader
        else:
            if 'test_loader' in self.trainer.ctx:
                del self.trainer.ctx['test_loader']

        if original_cur_split is not None:
            self.trainer.ctx.cur_split = original_cur_split
        else:
            if 'cur_split' in self.trainer.ctx:
                del self.trainer.ctx['cur_split']

        if original_data is not None and hasattr(original_data, '__getitem__'):
            try:
                if original_test_data is not None:
                    original_data['test'] = original_test_data
            except (TypeError, KeyError):
                pass

    def _log_domain_results(self, domain_results, domain_sizes):
        """Log domain-specific evaluation results and calculate weighted average."""
        logger.info("-"*80)
        logger.info("Domain-specific Test Results:")
        logger.info("-"*80)

        total_samples = sum(domain_sizes.values())

        # Collect all metric names from the first domain
        first_domain_metrics = next(iter(domain_results.values()))['metrics']
        metric_names = [key for key in first_domain_metrics.keys()
                      if key.startswith('test_') and not key.endswith('_total')]

        # Calculate weighted average for all metrics
        weighted_metrics = {}
        for metric_name in metric_names:
            weighted_value = 0.0
            for domain in domain_results.keys():
                result = domain_results[domain]
                size = result['size']
                weight = size / total_samples if total_samples > 0 else 0
                metric_value = result['metrics'].get(metric_name, 0.0)
                weighted_value += metric_value * weight
            weighted_metrics[metric_name] = weighted_value

        # Log accuracy for each domain
        for domain in sorted(domain_results.keys()):
            result = domain_results[domain]
            acc = result['accuracy']
            size = result['size']
            weight = size / total_samples if total_samples > 0 else 0

            if acc is not None:
                logger.info(f"  {domain:12s}: Acc = {acc:.4f} | Samples = {size:4d} | Weight = {weight:.3f}")
            else:
                logger.info(f"  {domain:12s}: Acc = N/A    | Samples = {size:4d} | Weight = {weight:.3f}")

        logger.info("-"*80)
        weighted_acc = weighted_metrics.get('test_acc', 0.0)
        logger.info(f"  {'Weighted Avg':12s}: Acc = {weighted_acc:.4f} | Total Samples = {total_samples}")
        logger.info("="*80)

        # Store weighted average in best_results
        weighted_results = {
            'test_total': total_samples,
        }
        # Add all weighted metrics
        weighted_results.update(weighted_metrics)

        # Format weighted average results
        formatted_weighted_res = self._monitor.format_eval_res(
            weighted_results,
            rnd=self.state,
            role='Server #Weighted_Avg',
            forms=self._cfg.eval.report,
            return_raw=True
        )

        # Update best results for weighted average (only if results are not empty)
        if formatted_weighted_res.get('Results_raw'):
            self._monitor.update_best_result(
                self.best_results,
                formatted_weighted_res['Results_raw'],
                results_type="server_global_eval_weighted"
            )
        else:
            logger.warning("Empty weighted evaluation results, skipping best result update")

        # Merge with history
        from federatedscope.core.auxiliaries.utils import merge_dict_of_results
        self.history_results = merge_dict_of_results(
            self.history_results, formatted_weighted_res
        )


def call_fedavg_domain_eval_worker(method: str):
    """
    Register worker builder for fedavg_domain_eval method.

    Args:
        method: The federate method name

    Returns:
        Dictionary with 'client' and 'server' classes, or None if method doesn't match
    """
    if method.lower() == 'fedavg_domain_eval':
        return {
            'client': Client,  # Use standard client for FedAvg
            'server': FedAvgDomainEvalServer,
        }
    return None


# Register the worker
register_worker('fedavg_domain_eval', call_fedavg_domain_eval_worker)


