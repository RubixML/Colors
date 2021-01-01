<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Clusterers\KMeans;
use Rubix\ML\Other\Loggers\Screen;
use Rubix\ML\CrossValidation\Reports\ContingencyTable;
use Rubix\ML\CrossValidation\Metrics\Homogeneity;
use Rubix\ML\Datasets\Unlabeled;

use function Rubix\ML\array_transpose;

ini_set('memory_limit', '-1');

$logger = new Screen();

$generator = new Agglomerate([
    'red' => new Blob([255, 0, 0], 20.0),
    'orange' => new Blob([255, 128, 0], 10.0),
    'yellow' => new Blob([255, 255, 0], 10.0),
    'green' => new Blob([0, 128, 0], 20.0),
    'blue' => new Blob([0, 0, 255], 20.0),
    'aqua' => new Blob([0, 255, 255], 10.0),
    'purple' => new Blob([128, 0, 255], 10.0),
    'pink' => new Blob([255, 0, 255], 10.0),
    'magenta' => new Blob([255, 0, 128], 10.0),
    'black' => new Blob([0, 0, 0], 10.0),
]);

$logger->info('Generating dataset');

[$training, $testing] = $generator->generate(5000)->stratifiedSplit(0.8);

$estimator = new KMeans(10);

$estimator->setLogger($logger);

$estimator->train($training);

$losses = $estimator->steps();

Unlabeled::build(array_transpose([$losses]))
    ->toCSV(['losses'])
    ->write('progress.csv');

$logger->info('Progress saved to progress.csv');

$logger->info('Making predictions');

$predictions = $estimator->predict($testing);

$report = new ContingencyTable();

$results = $report->generate($predictions, $testing->labels());

echo $results;

$results->toJSON()->write('report.json');

$logger->info('Report saved to report.json');

$metric = new Homogeneity();

$score = $metric->score($predictions, $testing->labels());

$logger->info('Clusters are ' . (string) round($score * 100.0, 2) . '% homogenous');
