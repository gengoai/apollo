package com.gengoai.apollo.ml.sequence;

import com.gengoai.Stopwatch;
import com.gengoai.apollo.linear.Axis;
import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.linear.NDArrayFactory;
import com.gengoai.apollo.linear.NDArrayInitializer;
import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.FitParameters;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.apollo.ml.preprocess.Preprocessor;
import com.gengoai.apollo.ml.preprocess.PreprocessorList;
import com.gengoai.apollo.ml.vectorizer.IndexVectorizer;
import com.gengoai.apollo.ml.vectorizer.Vectorizer;
import com.gengoai.apollo.optimization.TerminationCriteria;
import com.gengoai.collection.Iterables;
import com.gengoai.collection.Streams;
import com.gengoai.conversion.Cast;
import com.gengoai.function.SerializableSupplier;
import com.gengoai.stream.MStream;

import java.util.Iterator;

/**
 * The type Greedy avg perceptron.
 *
 * @author David B. Bracewell
 */
public class GreedyAvgPerceptron extends SequenceLabeler {
   private static final long serialVersionUID = 1L;
   private NDArray bias;
   private NDArray featureWeights;
   private NDArray transitionWeights;

   public GreedyAvgPerceptron(Preprocessor... preprocessors) {
      super(Validator.ALWAYS_TRUE,
            IndexVectorizer.featureVectorizer(),
            preprocessors);
   }

   public GreedyAvgPerceptron(Validator validator, Preprocessor... preprocessors) {
      super(validator,
            IndexVectorizer.featureVectorizer(),
            preprocessors);
   }


   public GreedyAvgPerceptron(Vectorizer<String> featureVectorizer, PreprocessorList preprocessors) {
      super(Validator.ALWAYS_TRUE, featureVectorizer, preprocessors);
   }

   public GreedyAvgPerceptron(Vectorizer<String> featureVectorizer, Preprocessor... preprocessors) {
      super(Validator.ALWAYS_TRUE, featureVectorizer, preprocessors);
   }

   public GreedyAvgPerceptron(Validator validator, Vectorizer<String> featureVectorizer, PreprocessorList preprocessors) {
      super(validator, featureVectorizer, preprocessors);
   }

   public GreedyAvgPerceptron(Validator validator, Vectorizer<String> featureVectorizer, Preprocessor... preprocessors) {
      super(validator, featureVectorizer, preprocessors);
   }


   @Override
   protected SequenceLabeler fitPreprocessed(Dataset preprocessed, FitParameters fitParameters) {
     return fit(() -> preprocessed.stream().map(this::encode), Cast.as(fitParameters, Parameters.class));
   }


   @Override
   public Labeling label(Example example) {
      String[] labels = new String[example.size()];
      int pLabel = (int) bias.length();
      NDArray sequence = encodeAndPreprocess(example);
      for (int i = 0; i < sequence.numRows(); i++) {
         NDArray distribution = distribution(sequence.getVector(i, Axis.ROW), pLabel);
         int cLabel = (int) distribution.argMax(Axis.ROW).get(0);
         boolean isValid = isValidTransition(cLabel, pLabel, example.getExample(i));
         while (!isValid) {
            distribution.set(cLabel, Double.NEGATIVE_INFINITY);
            cLabel = (int) distribution.argMax(Axis.ROW).get(0);
            isValid = isValidTransition(cLabel, pLabel, example.getExample(i));
         }
         labels[i] = getLabelVectorizer().decode(cLabel);
         pLabel = cLabel;
      }
      return new Labeling(labels);
   }

   private SequenceLabeler fit(SerializableSupplier<MStream<NDArray>> dataSupplier, Parameters parameters) {
      this.featureWeights = NDArrayFactory.SPARSE.create(NDArrayInitializer.rand,
                                                         getNumberOfFeatures(),
                                                         getNumberOfLabels());
      this.transitionWeights = NDArrayFactory.SPARSE.zeros(getNumberOfLabels() + 1,
                                                           getNumberOfLabels());
      this.bias = NDArrayFactory.SPARSE.zeros(getNumberOfLabels());

      final NDArray fTotals = featureWeights.copy();
      final NDArray tTotals = transitionWeights.copy();
      final NDArray bTotals = bias.copy();
      final NDArray fTimestamps = featureWeights.copy();
      final NDArray tTimestamps = transitionWeights.copy();
      final NDArray bTimestamps = bias.copy();

      int instances = 0;
      TerminationCriteria terminationCriteria = TerminationCriteria.create()
                                                                   .historySize(parameters.historySize)
                                                                   .maxIterations(parameters.maxIterations)
                                                                   .tolerance(parameters.eps);
      Stopwatch sw = Stopwatch.createStarted();
      for (int i = 0; i < terminationCriteria.maxIterations(); i++) {
         double total = 0;
         double correct = 0;

         for (NDArray sequence : dataSupplier.get().shuffle()) {
            int pLabel = getNumberOfLabels();
            NDArray y = sequence.getLabelAsNDArray().argMax(Axis.ROW);
            for (int row = 0; row < sequence.numRows(); row++) {
               final int predicted = (int) predict(sequence.getVector(row, Axis.ROW), pLabel);
               final int gold = (int) y.get(row);
               total++;
               if (predicted != gold) {
                  instances++;
                  for (NDArray.Entry e : Iterables.asIterable(sequence.sparseRowIterator(row))) {
                     update(gold, e.getColumn(), 1.0, instances, featureWeights, fTimestamps, fTotals);
                     update(predicted, e.getColumn(), -1.0, instances, featureWeights, fTimestamps, fTotals);
                  }
                  update(gold, pLabel, 1.0, instances, transitionWeights, tTimestamps, tTotals);
                  update(predicted, pLabel, -1.0, instances, transitionWeights, tTimestamps, tTotals);
                  updateBias(gold, 1.0, instances, bTimestamps, bTotals);
                  updateBias(predicted, -1.0, instances, bTimestamps, bTotals);
               } else {
                  correct++;
               }
               pLabel = gold;
            }
         }
         double acc = (correct / total) * 100;
         sw.stop();
         System.out.println(String.format("Iteration %d complete. %.3f accuracy (%s)", i + 1, acc, sw));
         sw.reset();
         if (terminationCriteria.check(100 - acc)) {
            break;
         }
         sw.start();
      }

      fTotals.addi(fTimestamps.rsub(instances).mul(featureWeights));
      tTotals.addi(tTimestamps.rsub(instances).mul(transitionWeights));
      bTotals.addi(bTimestamps.rsub(instances).mul(bias));
      featureWeights = fTotals.div(instances);
      transitionWeights = tTotals.div(instances);
      bias = bTotals.div(instances);
      return this;
   }

   private double dot(Iterator<NDArray.Entry> itr, NDArray other) {
      return Streams.asStream(itr)
                    .mapToDouble(e -> other.get(e.matrixIndex()) * e.getValue())
                    .sum();
   }

   @Override
   public Parameters getDefaultFitParameters() {
      return new Parameters();
   }


   /**
    * Custom fit parameters for the GreedyAveragePerceptron
    */
   public static class Parameters extends FitParameters {
      private static final long serialVersionUID = 1L;
      /**
       * The maximum number of iterations to run for
       */
      public int maxIterations = 100;
      /**
       * The epsilon to use for checking for convergence.
       */
      public double eps = 1e-5;

      /**
       * The number of iterations to use for determining convergence
       */
      public int historySize = 3;
   }


   private float predict(NDArray row, int pLabel) {
      return distribution(row, pLabel).argMax(Axis.ROW).get(0);
   }

   private NDArray distribution(NDArray row, int pLabel) {
      NDArray scores = row.mmul(featureWeights);
      scores.addi(transitionWeights.getVector(pLabel, Axis.ROW))
            .addi(bias);
      return scores;
   }

   private void update(int cls, int feature, double value, int iteration, NDArray weights, NDArray timeStamp, NDArray totals) {
      int iterAt = iteration - (int) timeStamp.get(feature, cls);
      totals.increment(feature, cls, iterAt * weights.get(feature, cls));
      weights.increment(feature, cls, value);
      timeStamp.set(feature, cls, iteration);
   }

   private void updateBias(int cls, double value, int iteration, NDArray timeStamp, NDArray totals) {
      int iterAt = iteration - (int) timeStamp.get(cls);
      totals.increment(cls, iterAt * bias.get(cls));
      bias.increment(cls, value);
      timeStamp.set(cls, iteration);
   }
}//END OF WindowSequenceLabeler
