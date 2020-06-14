from pyro.infer import MCMC, NUTS


class MCMCSampler:
    def __init__(self, model, kernel=NUTS, **options):
        self._model = model
        self._kernel = kernel
        self._defaults = {'num_samples': 100, 'warmup_steps': 0, 'jit_compile': True}.update(options)
        self._mcmc = None

    def run_mcmc(self, *args, initial_params=None, **options):
        params = self._defaults.copy()
        params.update(options)

        self._mcmc = MCMC(kernel=self._kernel(self._model,
                                              jit_compile=params['jit_compile'],
                                              ignore_jit_warnings=params['jit_compile']),
                          num_samples=params['num_samples'],
                          warmup_steps=params['warmup_steps'],
                          initial_params=initial_params)
        return self._mcmc.run(*args)

    @property
    def samples(self):
        if self._mcmc:
            return self._mcmc.get_samples()

    @property
    def sample_mean(self):
        if self._mcmc:
            return {latent: samples.mean(0) for latent, samples in self.samples.items()}
