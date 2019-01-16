use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;

type Pixels = Vec<i32>;

#[derive(Clone, Debug)]
struct Observation {
    label: String,
    pixels: Pixels,
}

fn create_observation(entry: String) -> Observation {
    let parts = entry.split(",").map(|e| e.to_string()).collect::<Vec<_>>();

    let label = parts.first().unwrap().clone();
    let pixels = parts
        .iter()
        .skip(1)
        .map(|e| e.parse::<i32>().unwrap())
        .collect::<Vec<_>>();

    Observation {
        label: label,
        pixels: pixels,
    }
}

fn read_observations(path: &str) -> Vec<Observation> {
    let f = File::open(path).unwrap();
    let file = BufReader::new(&f);

    file.lines()
        .skip(1)
        .map(|e| e.unwrap())
        .map(create_observation)
        .collect::<Vec<_>>()
}

fn manhattan_distance(pixels1: &Pixels, pixels2: &Pixels) -> i32 {
    pixels1
        .iter()
        .zip(pixels2)
        .map(|(p1, p2)| (p1 - p2).abs())
        .sum()
}

fn train<FDistance>(
    training_set: Vec<Observation>,
    distance: FDistance,
) -> impl Fn(&Pixels) -> String
where
    FDistance: Fn(&Pixels, &Pixels) -> i32 + 'static,
{
    move |pixels| {
        training_set
            .iter()
            .min_by_key(|obs| distance(&obs.pixels, pixels))
            .unwrap()
            .clone()
            .label
    }
}

fn evaluate<FClassifier>(validation_set: Vec<Observation>, classifier: FClassifier) -> f32
where
    FClassifier: Fn(&Pixels) -> String + 'static,
{
    let sum = validation_set
        .iter()
        .map(|obs| {
            let predicted = classifier(&obs.pixels);

            println!("Predicted: {}, Actual: {}", predicted, obs.label);

            if predicted == obs.label {
                1.
            } else {
                0.
            }
        })
        .sum::<f32>();

    sum / validation_set.len() as f32
}

fn main() {
    let training_path = "./trainingsample.csv";
    let validation_path = "./validationsample.csv";

    let training_data = read_observations(training_path);
    let validation_data = read_observations(validation_path);

    let manhattan_classifier = train(training_data, manhattan_distance);

    let correct_percent = evaluate(validation_data, manhattan_classifier);

    println!("Correct percent: {}%", correct_percent);
}
