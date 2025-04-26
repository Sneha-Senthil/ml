1. Bagging (Bootstrap Aggregating)
When to use:

When you have high-variance models (e.g., Decision Trees).

To reduce overfitting by averaging predictions from multiple models.

When you want a model that can be easily parallelized.

2. Boosting
When to use:

When you have high-bias models (e.g., weak learners).

When you want to improve accuracy by focusing on hard-to-predict instances.

When sequential improvement is needed, i.e., each model corrects the mistakes of the previous one.

3. Stacking
When to use:

When you have multiple diverse models that perform well individually.

To combine the strengths of different algorithms (e.g., trees, logistic regression, SVMs).

When you have complex datasets and no single model performs consistently best.

4. Voting
When to use:

When you have several strong models with complementary strengths.

When you want to combine predictions from multiple models using majority voting (or averaging for regression).

When you have a diverse set of classifiers, and want to improve overall performance by combining them.