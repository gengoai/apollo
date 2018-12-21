package com.gengoai.apollo.ml.classification;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.linear.NDArrayFactory;
import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.FitParameters;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.apollo.ml.preprocess.Preprocessor;
import com.gengoai.apollo.ml.preprocess.PreprocessorList;
import com.gengoai.apollo.ml.vectorizer.Vectorizer;
import com.gengoai.collection.Iterables;
import com.gengoai.conversion.Cast;

import java.util.Arrays;

import static com.gengoai.Validation.notNull;

/**
 * The type Naive bayes.
 *
 * @author David B. Bracewell
 */
public class NaiveBayes extends Classifier {
   private ModelType modelType;
   private double[] priors;
   private double[][] conditionals;

   /**
    * Instantiates a new Naive bayes.
    *
    * @param preprocessors the preprocessors
    */
   public NaiveBayes(Preprocessor... preprocessors) {
      super(preprocessors);
   }

   /**
    * Instantiates a new Naive bayes.
    *
    * @param labelVectorizer   the label vectorizer
    * @param featureVectorizer the feature vectorizer
    * @param preprocessors     the preprocessors
    */
   public NaiveBayes(Vectorizer<String> labelVectorizer, Vectorizer<String> featureVectorizer, Preprocessor... preprocessors) {
      super(labelVectorizer, featureVectorizer, preprocessors);
   }

   /**
    * Instantiates a new Naive bayes.
    *
    * @param labelVectorizer   the label vectorizer
    * @param featureVectorizer the feature vectorizer
    * @param preprocessors     the preprocessors
    */
   public NaiveBayes(Vectorizer<String> labelVectorizer, Vectorizer<String> featureVectorizer, PreprocessorList preprocessors) {
      super(labelVectorizer, featureVectorizer, preprocessors);
   }

   /**
    * Instantiates a new Naive bayes.
    *
    * @param featureVectorizer the feature vectorizer
    * @param preprocessors     the preprocessors
    */
   public NaiveBayes(Vectorizer<String> featureVectorizer, Preprocessor... preprocessors) {
      super(featureVectorizer, preprocessors);
   }

   /**
    * Instantiates a new Naive bayes.
    *
    * @param featureVectorizer the feature vectorizer
    * @param preprocessors     the preprocessors
    */
   public NaiveBayes(Vectorizer<String> featureVectorizer, PreprocessorList preprocessors) {
      super(featureVectorizer, preprocessors);
   }

   @Override
   protected void fitPreprocessed(Dataset preprocessed, FitParameters fitParameters) {
      Parameters parameters = notNull(Cast.as(fitParameters, Parameters.class));
      conditionals = new double[getNumberOfFeatures()][getNumberOfLabels()];
      priors = new double[getNumberOfLabels()];
      modelType = parameters.modelType;
      double[] labelCounts = new double[getNumberOfLabels()];

      double N = 0;
      for (Example instance : preprocessed) {
         if (instance.hasLabel()) {
            N++;
            NDArray vector = encode(instance);
            int ci = vector.getLabelAsNDArray().argMax();
            priors[ci] += instance.getWeight();
            for (NDArray.Entry entry : Iterables.asIterable(vector.sparseIterator())) {
               labelCounts[ci] += entry.getValue();
               conditionals[(int) entry.getIndex()][ci] += instance.getWeight() * modelType.convertValue(
                  entry.getValue());
            }
         }
      }

      double V = getNumberOfFeatures();
      for (int featureIndex = 0; featureIndex < conditionals.length; featureIndex++) {
         double[] tmp = Arrays.copyOf(conditionals[featureIndex], conditionals[featureIndex].length);
         for (int labelIndex = 0; labelIndex < priors.length; labelIndex++) {
            if (modelType == NaiveBayes.ModelType.Complementary) {
               double nCi = 0;
               double nC = 0;
               for (int j = 0; j < priors.length; j++) {
                  if (j != labelIndex) {
                     nCi += tmp[j];
                     nC += labelCounts[j];
                  }
               }
               conditionals[featureIndex][labelIndex] = Math.log(
                  modelType.normalize(nCi, priors[labelIndex], nC, V));
            } else {
               conditionals[featureIndex][labelIndex] = Math.log(
                  modelType.normalize(conditionals[featureIndex][labelIndex], priors[labelIndex],
                                      labelCounts[labelIndex], V));
            }
         }

      }

      for (int i = 0; i < priors.length; i++) {
         priors[i] = Math.log(priors[i] / N);
      }
   }

   @Override
   public FitParameters getDefaultFitParameters() {
      return new Parameters();
   }

   @Override
   public Classification predict(Example example) {
      return new Classification(modelType.distribution(encodeAndPreprocess(example),
                                                       priors,
                                                       conditionals),
                                getLabelVectorizer());
   }


   /**
    * The type Parameters.
    */
   public static class Parameters extends FitParameters {
      /**
       * The type.
       */
      public ModelType modelType = ModelType.Bernoulli;
   }

   /**
    * Three types of Naive Bayes  are supported each of which have their own potential pros and cons and may work better
    * or worse for different types of data.
    */
   public enum ModelType {
      /**
       * Multinomial Naive Bayes using Laplace Smoothing
       */
      Multinomial,
      /**
       * Bernoulli Naive Bayes where each feature is treated as being binary
       */
      Bernoulli {
         @Override
         double convertValue(double value) {
            return value > 0 ? 1.0 : 0.0;
         }

         @Override
         double normalize(double conditionalCount, double priorCount, double totalLabelCount, double V) {
            return (conditionalCount + 1) / (priorCount + 2);
         }

         @Override
         NDArray distribution(NDArray instance, double[] priors, double[][] conditionals) {
            NDArray distribution = NDArrayFactory.columnVector(priors);
            for (int i = 0; i < priors.length; i++) {
               for (int f = 0; f < conditionals.length; f++) {
                  if (instance.get(f) != 0) {
                     distribution.increment(i, Math.log(conditionals[f][i]));
                  } else {
                     distribution.increment(i, Math.log(1 - conditionals[f][i]));
                  }
               }
            }
            distribution.mapi(Math::exp);
            return distribution;
         }
      },
      /**
       * Complementary Naive Bayes which works similarly to the Multinomial version, but is trained differently to
       * better handle label imbalance.
       */
      Complementary;

      /**
       * Converts a features value.
       *
       * @param value the value
       * @return the converted value
       */
      double convertValue(double value) {
         return value;
      }

      /**
       * Calculates a distribution of probabilities over the labels given a vector instance and the priors and
       * conditionals.
       *
       * @param instance     the instance to calculate the distribution for
       * @param priors       the label priors
       * @param conditionals the feature-label conditional probabilities
       * @return the distribution as an array
       */
      NDArray distribution(NDArray instance, double[] priors, double[][] conditionals) {
         NDArray distribution = NDArrayFactory.columnVector(priors);
         instance.forEachSparse(entry -> {
            for (int i = 0; i < priors.length; i++) {
               distribution.decrement(i, entry.getValue() * conditionals[(int) entry.getIndex()][i]);
            }
         });
         return distribution.mapi(Math::exp);
      }

      /**
       * Normalizes (smooths) the conditional probability given the conditional count, prior count, total label count,
       * and vocabulary size.
       *
       * @param conditionalCount the conditional count
       * @param priorCount       the prior count
       * @param totalLabelCount  the total label count
       * @param V                the vocabulary size
       * @return the normalized  (smoothed) conditional probability
       */
      double normalize(double conditionalCount, double priorCount, double totalLabelCount, double V) {
         return (conditionalCount + 1) / (totalLabelCount + V);
      }

   }
}//END OF NaiveBayes
