<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\Clusterers\KMeans;
use Rubix\ML\Other\Loggers\Screen;
use Rubix\ML\CrossValidation\Reports\ContingencyTable;
use League\Csv\Writer;

use function Rubix\ML\array_transpose;

ini_set('memory_limit', '-1');

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

[$training, $testing] = $generator->generate(5000)->stratifiedSplit(0.8);

$estimator = new KMeans(10);

$estimator->setLogger(new Screen('colors'));

$estimator->train($training);

$losses = $estimator->steps();

$writer = Writer::createFromPath('progress.csv', 'w+');
$writer->insertOne(['loss']);
$writer->insertAll(array_transpose([$losses]));

echo 'Progress saved to progress.csv' . PHP_EOL;

$predictions = $estimator->predict($testing);

$report = new ContingencyTable();

$results = $report->generate($predictions, $testing->labels());

file_put_contents('report.json', json_encode($results, JSON_PRETTY_PRINT));

echo 'Report saved to report.json' . PHP_EOL;