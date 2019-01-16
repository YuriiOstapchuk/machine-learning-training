import * as fs from 'fs';
import * as util from 'util';

type Pixels = number[];

type Observation = {
  label: string;
  pixels: Pixels;
};

type Distance = (pixels1: Pixels, pixels2: Pixels) => number;

type Classifier = (pixels: Pixels) => string;

const createObservation = (entry: string): Observation => {
  const [label, ...rawPixels] = entry.split(',');
  const pixels = rawPixels.map(Number);

  return {
    label,
    pixels,
  };
};

const readObservations = (path: string) => {
  const readFile = util.promisify(fs.readFile);

  return readFile(path, { encoding: 'utf8' })
    .then(data => data.toString().split('\n'))
    .then(([, ...observations]) =>
      observations.map(createObservation).filter(e => !!e.label),
    );
};

const zip = <A, B>(l1: A[], l2: B[]) => l1.map((e, i) => [e, l2[i]]);

const minBy = <T>(list: T[], f: (item: T) => number | string) =>
  list.reduce((acc, item) => (f(item) < f(acc) ? item : acc));

const sum = (a: number, b: number) => a + b;

const manhattanDistance: Distance = (pixels1, pixels2) =>
  zip(pixels1, pixels2)
    .map(([p1, p2]) => Math.abs(p1 - p2))
    .reduce(sum);

const train = (trainingSet: Observation[], distance: Distance) => (
  pixels: Pixels,
) =>
  minBy(trainingSet, observation => distance(observation.pixels, pixels)).label;

const evaluate = (validationSet: Observation[], classifier: Classifier) =>
  validationSet
    .map(
      (observation): number => {
        const predicted = classifier(observation.pixels);

        console.log(`Predicted: ${predicted}, Actual: ${observation.label}`);

        return predicted === observation.label ? 1 : 0;
      },
    )
    .reduce(sum) / validationSet.length;

const main = async () => {
  const trainingPath = './trainingsample.csv';
  const validationPath = './validationsample.csv';

  const trainingData = await readObservations(trainingPath);
  const validationData = await readObservations(validationPath);

  const manhattanClassifier = train(trainingData, manhattanDistance);

  const correctPercent = evaluate(validationData, manhattanClassifier);

  console.log(`Correct: ${correctPercent}`);
};

main();
