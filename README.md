# Color Clusterer
An unsupervised learning problem that involves clustering 10 different base colors generated on the fly using Rubix ML [Generators](https://docs.rubixml.com/en/latest/datasets/generators/api.html). Our objective is to generate a synthetic training and testing set that we'll use to train and test a [Gaussian Mixture](https://docs.rubixml.com/en/latest/clusterers/gaussian-mixture.html) clusterer.

- **Difficulty**: Easy
- **Training time**: < 1 Minute
- **Memory needed**: < 1G

## Installation
Clone the repository locally using [Git](https://git-scm.com/):
```sh
$ git clone https://github.com/RubixML/Colors
```

Install dependencies using [Composer](https://getcomposer.org/):
```sh
$ composer install
```

## Requirements
- [PHP](https://php.net) 7.1.3 or above

## Tutorial

### Introduction
In machine learning, synthetic data are often used to test an estimator and to augment a small dataset with more training data. In this tutorial we'll use synthetic data to train and test a Gaussian Mixture Model (GMM) to determine the color of a sample.

> **Note:** The source code for this example can be found in the [train.php](https://github.com/RubixML/Colors/blob/master/train.php) file in project root.

### Generating the Data
Rubix ML provides a number of [Generators](https://docs.rubixml.com/en/latest/datasets/generators/api.html) which can output a dataset in a particular shape and dimensionality. For this example project, we are going to generate [Blobs](https://docs.rubixml.com/en/latest/datasets/generators/blob.html) of colors using their red, green, and blue (RGB) values as the features. The [Agglomerate](https://docs.rubixml.com/en/latest/datasets/generators/agglomerate.html) generator will combine and label the individual color generators to form a dataset consisting of all 10 colors.

```php
use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\Datasets\Generators\Blob;

$generator = new Agglomerate([
    'red' => new Blob([255, 0, 0], 20.),
    'orange' => new Blob([255, 128, 0], 20.),
    'yellow' => new Blob([255, 255, 0], 20.),
    'green' => new Blob([0, 128, 0], 20.),
    'blue' => new Blob([0, 0, 255], 20.),
    'aqua' => new Blob([0, 255, 255], 20.),
    'purple' => new Blob([128, 0, 255], 20.),
    'pink' => new Blob([255, 0, 255], 20.),
    'magenta' => new Blob([255, 0, 128], 20.),
    'black' => new Blob([0, 0, 0], 20.),
]);
```

To generate the dataset, simply call `generate()` with the number of samples (*n*) to be generated. A [Dataset](https://docs.rubixml.com/en/latest/datasets/generators/api.html) object is returned which allows you to fluently process the data further if needed. For example we could stratify and split the dataset into a training and testing set such that each color is represented fairly in each set. The proportion of samples in the *left* (training) set to the *right* (testing) set is given by the *ratio* parameter to the `stratifiedSplit()` method. For this example, we'll choose to generate a set of 5,000 samples and then split it 80/20 (4000 for training and 1000 for testing).

```php
[$training, $testing] = $generator->generate(5000)->stratifiedSplit(0.8);
```

Now let's take a look at the data we've generated using some plotting software such as [Plotly](https://plot.ly).

![Synthetic Color Data](https://github.com/RubixML/Colors/blob/master/docs/images/samples-3d.png)

### Instantiating the Learner
Next we'll define our [Gaussian Mixture](https://docs.rubixml.com/en/latest/clusterers/gaussian-mixture.html) clusterer. Gaussian Mixture Models (GMMs) are a type of probabilistic model for finding subpopulations within a dataset. GMM places a Gaussian *component* over each target cluster that allows a likelihood function to be computed. The learner is then trained with the Expectation Maximization (EM) algorithm to maximize the likelihood that the area over each Gaussian component contains only samples of the same class. The number of target clusters (k) is passes as a hyper-parameter to the learner. In this case, we already know that the number of clusters should be 10 since we generated 10 color blobs.

```php
use Rubix\ML\Clusterers\GaussianMixture;

$estimator = new GaussianMixture(10);
```

### Training
Once the learner is instantiated, call the `train()` method passing in the training set we generated earlier.

```php
$estimator->train($training);
```

### Inference
To make the predictions, pass the testing set to the `predict()` method on the estimator instance. 
```php
$predictions = $estimator->predict($testing);
```

### Cross Validation
Lastly, to test the model, we'll create a report that compares the predictions to some ground truth given by the labels we've assigned. A [Contingency Table](https://docs.rubixml.com/en/latest/cross-validation/reports/contingency-table.html) is a clustering report similar to a [Confusion Matrix](https://docs.rubixml.com/en/latest/cross-validation/reports/confusion-matrix.html) but for clustering instead of classification. It counts the number of times a particular cluster was assigned to a given label. A good clustering should show that each cluster contains samples with roughly the same label.

We'll need the predictions made earlier as well as the labels from the testing set to pass to the report's `generate()` method. Then, we'll save the output to a JSON file so we can review it later.

```php
use Rubix\ML\CrossValidation\Reports\ContingencyTable;

$report = new ContingencyTable();

$results = $report->generate($predictions, $testing->labels());
```

Here is an excerpt of a Contingency Report that demonstrates a misclustered magenta point within the red cluster.

```json
{
    "8": {
        "red": 100,
        "orange": 0,
        "yellow": 0,
        "green": 0,
        "blue": 0,
        "aqua": 0,
        "purple": 0,
        "pink": 0,
        "magenta": 1,
        "black": 0
    },
}
```

### Wrap Up
- Clustering is a type of unsupervised learning which aims at finding samples which are simila to each other
- Synthetic data can be used as a way to test models or augment small datasets
- Generators are used to generate synthetic data in various shapes and dimensionalities
- A Guassian Mixture model is a type of probabilistic clusterer
- A Contingnecy Table is a report that allows you to evaluate a clusterer's generalization performance

### Next Steps
Try generating some more data in other shapes such a [Circle](https://docs.rubixml.com/en/latest/datasets/generators/circle.html) or [Half Moon](https://docs.rubixml.com/en/latest/datasets/generators/half-moon.html). See if Gaussian Mixture is able to detect clusters of different shapes and sizes.