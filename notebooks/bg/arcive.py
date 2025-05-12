# MT
mt_pred = []
for restart in range(num_restarts):
    print(f"{restart=}")
    train_x = initial_train_x.clone()
    for step in range(num_optimization_rounds):
        # fit the model 
        train_y = problem(train_x)
        train_y = botorch.utils.standardize(train_y)
        model = models.KroneckerMultiTaskGP(
            train_X = train_x,
            train_Y = train_y,
        )
        # Fit and train the model
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
        fit.fit_gpytorch_mll_scipy(mll)
        model.eval()
        # with torch.no_grad():
        #     weights = torch.ones(train_y.shape[-1], dtype = torch.double)
        #     trained_pred_dist = model.posterior(
        #         X = test_x,
        #         # posterior_transform = botorch.acquisition.ScalarizedPosteriorTransform(weights)
        #     )
        #     score = negative_log_predictive_density(trained_pred_dist.mvn, test_y)

        acf = botorch.acquisition.multi_objective.ExpectedHypervolumeImprovement(model,
                                                                                 ref_point = problem._ref_point,
                                                                                 partitioning = NondominatedPartitioning(
                                                                                     ref_point,
                                                                                     train_y
                                                                                 )
                                                                                 )
        new_candidate, acf_value = botorch.optim.optimize_acqf(
            acf, bounds = torch.tensor(problem._bounds).T, q = 1, num_restarts = 1, raw_samples = 64)
        train_x = torch.concat([train_x, new_candidate])

        best_f = train_y.sum(-1, keepdim = True).max()
        mt_pred.append({
            "score": float(score.detach().cpu().numpy()),
            "candidate": new_candidate.detach().cpu().numpy(),
            "best_f": float(best_f.max().detach().cpu().numpy()),
            "model": "MultiTask",
            "step": step,
            "restart": restart
        })


##

import botorch.acquisition.multi_objective
from botorch.models.transforms.outcome import Standardize
from botorch.utils.multi_objective.box_decompositions import (
    FastNondominatedPartitioning,
)

# Train two independent GP to model the tasks.


mt_pred = []

for restart in range(num_restarts):
    train_x = initial_train_x.clone()
    print(f"{restart=}")
    for step in range(num_optimization_rounds):
        # fit the model
        train_y = problem(train_x)
        model = models.KroneckerMultiTaskGP(
            train_X = train_x.unsqueeze(0),
            train_Y = train_y.unsqueeze(0),
            outcome_transform = Standardize(m = 2, batch_shape = torch.Size([1])),
            input_transform = botorch.models.transforms.input.Normalize(problem.dim, bounds = problem.bounds)
        )
        # Fit and train the model
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
        fit.fit_gpytorch_mll_scipy(mll)

        model.eval()

        with torch.no_grad():
            pred = model.posterior(train_x).mean
        partitioning = FastNondominatedPartitioning(
            ref_point = problem.ref_point,
            Y = pred,
        )

        acf = botorch.acquisition.multi_objective.ExpectedHypervolumeImprovement(model,
                                                                                 ref_point = problem._ref_point,
                                                                                 partitioning = partitioning,
                                                                                 sampler = SobolQMCNormalSampler(
                                                                                     sample_shape = torch.Size(
                                                                                         [MC_SAMPLES])))
        new_candidate, _ = botorch.optim.optimize_acqf(
            acf, bounds = standard_bounds, q = 1, num_restarts = 10, raw_samples = 512)

        new_candidate = botorch.utils.transforms.unnormalize(new_candidate, problem.bounds).detach()
        train_x = torch.concat([train_x, new_candidate])
        new_y = problem(new_candidate)

        best_f = train_y.sum(-1, keepdim = True).max()
        bd = DominatedPartitioning(ref_point = problem.ref_point, Y = train_y)
        volume = bd.compute_hypervolume().item()
        # volume = partitioning.compute_hypervolume().item()
        mt_pred.append({
            "score": float(volume),
            "branin": float(new_y[:, 0].cpu().detach().numpy()),
            "currin": float(new_y[:, 1].cpu().detach().numpy()),
            "candidate": new_candidate.detach().cpu().numpy(),
            "best_f": float(best_f.detach().cpu().numpy()),
            "model": "MultiTask",
            "step": step,
            "restart": restart,
        })