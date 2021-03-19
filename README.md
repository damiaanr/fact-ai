# General instructions

Please be aware that we have two different versions of the code belonging to this project:

1. The originally provided code by Plumb. et al, upgraded to TensorFlow 2.x
    1. For the (by Plumb et al.) pre-trained models that are simply _evaluated_, head to `ELDR-TF2.x_(pre_trained_models)` for further instructions
    2. For the (by us) re-trained models, using the upgraded existing code, head to `ELDR-TF2.x_(newly_trained_models)` for further instructions
2. Our new, from-scratch, implementation which includes additional dimensionality reduction algorithms and datasets - head to `ELDR-NEW`

## Results

We have been able to successfully reproduce experiments and results by running the pre-loaded models, after re-training these models, and by rewriting the algorithm from scratch. Concrete performance results are provided within the readme of the `ELDR-NEW` section.


## Dataset information


|                           | Seeds                      | Glass          | Wine            | Heart                     | Iris           | Housing        | RNA             |
| ------                    | ----                       | -----          | -----           | ----                      | ----           | -----          | ---             |
| Data Set Characteristics  | Multivariate               | Multivariate   | Multivariate    | Multivariate              | Multivariate   | Multivariate   | Multivariate    |
| Attribute Characteristics | Real                       | Real           | Integer, Real   | Categorical, Integer, Real| Real           | Real           | Real            |
| Associated Tasks          | Classification, Clustering | Classification | Classification  | Classification            | Classification | Classification | Classification  |
| Number of Instances       | 210                        | 214            | 178             | 303                       | 150            | 506            | 25000           |
| Number of Attributes      | 7                          | 13             | 10              | 75*                       | 4              | 13             | 13166           |
| Missing Values            | N/A                        | No             | No              | Yes                       | No             | No             | No              |
| Area                      | Life                       | Physical       | Physical        | Life                      | Life           | Physical       | Life            |

*only 14 attributes used
### Source

Housing: https://archive.ics.uci.edu/ml/machine-learning-databases/housing/

Iris: https://archive.ics.uci.edu/ml/datasets/iris

Heart: https://archive.ics.uci.edu/ml/datasets/heart+disease

Seeds: https://archive.ics.uci.edu/ml/datasets/seeds 

Glass: https://archive.ics.uci.edu/ml/datasets/glass+identification

Wine:  https://archive.ics.uci.edu/ml/datasets/wine
