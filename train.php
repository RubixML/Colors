<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\Other\Loggers\Screen;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Clusterers\GaussianMixture;
use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\CrossValidation\Reports\ContingencyTable;

echo '╔═══════════════════════════════════════════════════════════════╗' . PHP_EOL;
echo '║                                                               ║' . PHP_EOL;
echo '║ Color Clusterer using Gaussian Mixture                        ║' . PHP_EOL;
echo '║                                                               ║' . PHP_EOL;
echo '╚═══════════════════════════════════════════════════════════════╝' . PHP_EOL;
echo PHP_EOL;

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

$estimator = new GaussianMixture(10);

$estimator->setLogger(new Screen('colors'));

$estimator->train($training);

$predictions = $estimator->predict($testing);

$report = new ContingencyTable();

$results = $report->generate($predictions, $testing->labels());

file_put_contents('report.json', json_encode($results, JSON_PRETTY_PRINT));

echo 'Report saved to report.json' . PHP_EOL;