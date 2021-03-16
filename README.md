# General instructions

Please be aware that we have two different versions of the code belonging to this project:

1. The originally provided code by Plumb. et al, upgraded to TensorFlow 2.x
    1. For the (by Plumb et al.) pre-trained models that are simply _evaluated_, head to `ELDR-TF2.x_(pre_trained_models)` for further instructions
    2. For the (by us) re-trained models, using the upgraded existing code, head to `ELDR-TF2.x_(newly_trained_models)` for further instructions
2. Our new, from-scratch, implementation which includes additional dimensionality reduction algorithms and datasets - head to `ELDR-NEW`

## Dataset information


|                           | Seeds                      | Glass          | Wine            |
| ------                    | ----                       | -----          | -----           |
| Data Set Characteristics: | Multivariate               | Multivariate   | Multivariate    |
| Attribute Characteristics:| Real                       | Real           | Integer, Real   |
| Associated Tasks:         | Classification, Clustering | Classification | Classification  |
| Number of Attributes:     | 7                          | 13             | 10              |
| Missing Values            | N/A                        | No             | No              |
| Area                      | Life                       | Physical       | Physical        |

### Source

Seeds: https://archive.ics.uci.edu/ml/datasets/seeds 

Glass: https://archive.ics.uci.edu/ml/datasets/glass+identification

Wine:  https://archive.ics.uci.edu/ml/datasets/wine
