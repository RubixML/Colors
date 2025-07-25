<?php
ini_set('display_errors', 1);
ini_set('display_startup_errors', 1);
error_reporting(E_ALL);

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Clusterers\KMeans;
use Rubix\ML\Pipeline;
use Rubix\ML\Transformers\OneHotEncoder;
use Rubix\ML\Transformers\ZScaleStandardizer;
use Rubix\ML\Loggers\Screen;

$vocabulary = ['aardvark', 'she', 'it', 'fancy', 'zebra'];


$zeros = array_fill(0, count($vocabulary), 0.0);
$template = array_combine($vocabulary, $zeros);
$samples = [];

$users  =       array(
                        array(  'customer_id'   =>  1,  'age'   =>  18,   'level'   =>  1  ,'learnedWords'  =>   array('aardvark', 'she')   ),
                        array(  'customer_id'   =>  2,  'age'   =>  28,   'level'   =>  2  ,'learnedWords'  =>   array('she', 'it', 'fancy')   )
);

foreach ($users as $user) {
   $sample = [];

   $sample[] = $user['age'];
   $sample[] = $user['level'];

   $vector = $template;

   foreach ($user['learnedWords'] as $word) {
       $vector[$word] = 1.0;
   }

  $vector = array_values($vector);

  $sample = array_merge($sample, $vector);

  $samples[] = $sample;
}

//echo '<pre>';var_dump($samples);echo '</pre>';
//die();
//normalize those vectors so their sum adds exactly to 1 (to remove the bias of new vs veteran users)

$training = new Unlabeled($samples);

$estimator = new KMeans(2, 128, 1000); // 50 target clusters, process 128 samples at a time, train for max 1000 epochs);

$estimator->setLogger(new Screen()); // This will output real-time training info to the console

$estimator = new Pipeline([
    new OneHotEncoder(), // this is to convert the categorical features (ex. sex, education-level, etc.) to numerical
    new ZScaleStandardizer(), // this is to ensure that all features use the same scale
], $estimator);

$estimator->train($training);

$predictions = $estimator->predict($training);

var_dump($predictions); // [20, 10, 34, 50, 2, 10, ... 49];
/*
foreach ($users as $i => $user) {
    $user->clusterNumber = $predictions[$i];
    
    $user->save();
}
*/
?>