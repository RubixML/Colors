# Color Clusterer

An *unsupervised* learning problem that involves clustering similar shades of 10 different base colors generated on the fly using Rubix [Generators](https://docs.rubixml.com/en/latest/datasets/generators/api.html). The objective is to generate a training and testing set full of synthetic data that we'll later use to train and test a [Gaussian Mixture](https://docs.rubixml.com/en/latest/clusterers/gaussian-mixture.html) clusterer. In this tutorial, you'll learn the concepts of unsupervised clustering and synthetic data generation.

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
In machine learning, synthetic data are used to either test an estimator or to augment a small dataset with more training data. Rubix provides a number of [Generators](https://docs.rubixml.com/en/latest/datasets/generators/api.html) which output a dataset in a particular shape and dimensionality. For this example project, we are going to generate [Blobs](https://docs.rubixml.com/en/latest/datasets/generators/blob.html) of colors using their RGB values as features. We'll form an [Aglomerate](https://docs.rubixml.com/en/latest/datasets/generators/agglomerate.html) of color Blobs and give each one a label corresponding to its base color name.

> **Note**: Generators can generate both labeled and unlabeled datasets. The type of Dataset object returned depends on the generator. See the [API Reference](https://docs.rubixml.com/en/latest/datasets/generators/api.html) for more details.

> The source code can be found in the [train.php](https://github.com/RubixML/Colors/blob/master/train.php) file in project root.

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

To generate a dataset, call `generate()` with the number of samples (*n*). A Dataset object is returned which allows you to fluently process the data further by *stratifying* and splitting the dataset into a training and testing set. Stratifying the dataset before splitting creates balanced training and testing sets by label. The proportion of samples in the *left* (training) set to the *right* (testing) set is given by the *ratio* parameter to the `stratifiedSplit()` method. Let's choose to generate a set of 5000 samples and then split it 80/20 (4000 for training and 1000 for testing).

```php
[$training, $testing] = $generator->generate(5000)->stratifiedSplit(0.8);
```

Let's take a look at the data we've just generated using plotting software such as [Plotly](https://plot.ly/). We've used the label to color the data such that each point is represented by its base color.

![Synthetic Color Data](https://github.com/RubixML/Colors/blob/master/docs/images/samples-3d.png)

Now we'll define our [Gaussian Mixture](https://docs.rubixml.com/en/latest/clusterers/gaussian-mixture.html) clusterer. Gaussian Mixture Models (*GMMs*) are a type of probabilistic model for finding subpopulations within a dataset. They place a Gaussian *component* over each target cluster that allows a *likelihood* function to be computed. The learner is then trained with Expectation Maximization (*EM*) to maximize the likelihood that the area over each Gaussian component contains only samples of the same class. To set the target number of clusters *k* we need to set the *hyper-parameters* of the GMM. Since we already know the number of different labeled color Blobs in our dataset we'll choose a value of 10.

```php
use Rubix\ML\Clusterers\GaussianMixture;

$estimator = new GaussianMixture(10);
```

Once our estimator is instantiated we can call `train()` passing in the training set we generated earlier.

```php
$estimator->train($training);
```

Lastly to test the model, let's create a report that compares the clustering to some ground truth given by the labels we've assigned to each Blob. A [Contingency Table](https://docs.rubixml.com/en/latest/cross-validation/reports/contingency-table.html) is a clustering report similar to a [Confusion Matrix](https://docs.rubixml.com/en/latest/cross-validation/reports/confusion-matrix.html). It counts the number of times a particular label was assigned to a cluster. A good clustering will show that each cluster contains samples with roughly the same label.

We'll need the predictions made by the Gaussian Mixture clusterer as well as the labels from the testing set to pass to the Contingency Table report's `generate()` method. Once that's done, we'll save the output to a JSON file so we can review it later.

```php
use Rubix\ML\CrossValidation\Reports\ContingencyTable;

$predictions = $estimator->predict($testing);

$report = new ContingencyTable();

$results = $report->generate($predictions, $testing->labels());
```

Here is an example of a cluster that contains a misclustered magenta point with the reds.

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

To run the training script from the project root:
```sh
$ php train.php
```

### Wrap Up

- Clustering is a type of *unsupervised* learning which aims at predicting the cluster label of a sample
- A Guassian Mixture model is a type of clusterer
- Synthetic data can be used as a way to test models or augment small datasets
- Rubix Generators are used to generate synthetic data in various shapes and dimensionalities
- A Contingnecy Table is a report that allows you to evaluate a clustering