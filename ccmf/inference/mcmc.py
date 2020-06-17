from pyro.infer import MCMC, NUTS


class MCMCSampler:
    """Class for MCMC sampling.

    """
    def __init__(self, model, kernel=NUTS, **options):
        self.__model = model
        self._kernel = kernel
        self.__defaults = {'num_samples': 100, 'warmup_steps': 0, 'jit_compile': True}
        self.__defaults.update(options)
        self._mcmc = None

    def run_mcmc(self, X, initial_params=None, **options):
        params = self.__defaults.copy()
        params.update(options)

        self._mcmc = MCMC(kernel=self._kernel(self.__model,
                                              jit_compile=params['jit_compile'],
                                              ignore_jit_warnings=params['jit_compile']),
                          num_samples=params['num_samples'],
                          warmup_steps=params['warmup_steps'],
                          initial_params=initial_params, num_chains=1)
        return self._mcmc.run(X)

    def get_samples(self):
        if self._mcmc:
            return self._mcmc.get_samples()

    @property
    def sample_mean(self):
        """Calculate mean of posterior samples.

        Returns
        -------

        """
        if self._mcmc:
            return {latent: samples.mean(0) for latent, samples in self.get_samples().items()}
