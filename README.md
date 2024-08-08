# HousePriceClassification
An house price classification task, where LDA and DBSCAN are combined to perform crucial Intraclass outlier detection.
A first analisys underlined a difficulty to discrimante successive price classes; after noticing that in the LDA (Linear Discriminant Analisys) space those classes are distributed as adjacent clusters,
the application of DBSCAN algorithm to the single classes, aimng to eliminate those records lying in the region between two clusters resulted in an effective method, enchances the Precision of the algorithms tested.
A well reasoned and statistically rigorous feature selection process has been also executed on the huge set of categorical and continuous feature sets.
