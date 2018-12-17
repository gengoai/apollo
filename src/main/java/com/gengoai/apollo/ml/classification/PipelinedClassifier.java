package com.gengoai.apollo.ml.classification;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.ml.*;
import com.gengoai.apollo.ml.preprocess.Preprocessor;
import com.gengoai.apollo.ml.vectorizer.BinaryVectorizer;
import com.gengoai.apollo.ml.vectorizer.IndexVectorizer;
import com.gengoai.apollo.ml.vectorizer.Vectorizer;
import com.gengoai.conversion.Cast;
import com.gengoai.function.SerializableSupplier;
import com.gengoai.stream.MStream;

/**
 * <p>Wraps a {@link Classifier} allowing it to work directly with {@link Dataset}s and {@link Example}s instead of
 * NDArray</p>
 *
 * @author David B. Bracewell
 */
public class PipelinedClassifier extends PipelinedModel implements Classifier {
   private static final long serialVersionUID = 1L;
   private final Classifier classifier;


   /**
    * Creates a {@link PipelinedClassifier} for binary problems
    *
    * @param classifier        the classifier
    * @param trueLabel         the true label
    * @param falseLabel        the false label
    * @param featureVectorizer the feature vectorizer
    * @param preprocessors     the preprocessors
    * @return the pipelined classifier
    */
   public static PipelinedClassifier binary(Classifier classifier,
                                            String trueLabel,
                                            String falseLabel,
                                            Vectorizer<String> featureVectorizer,
                                            Preprocessor... preprocessors
                                           ) {
      return new PipelinedClassifier(classifier,
                                     new BinaryVectorizer(trueLabel, falseLabel),
                                     featureVectorizer,
                                     preprocessors);
   }

   /**
    * Creates a {@link PipelinedClassifier} for binary problems with labels "true" and "false".
    *
    * @param classifier        the classifier
    * @param featureVectorizer the feature vectorizer
    * @param preprocessors     the preprocessors
    * @return the pipelined classifier
    */
   public static PipelinedClassifier binary(Classifier classifier,
                                            Vectorizer<String> featureVectorizer,
                                            Preprocessor... preprocessors
                                           ) {
      return new PipelinedClassifier(classifier,
                                     new BinaryVectorizer(),
                                     featureVectorizer,
                                     preprocessors);
   }

   /**
    * Creates a {@link PipelinedClassifier} for multi-class problems
    *
    * @param classifier        the classifier
    * @param featureVectorizer the feature vectorizer
    * @param preprocessors     the preprocessors
    * @return the pipelined classifier
    */
   public static PipelinedClassifier multiClass(Classifier classifier,
                                                Vectorizer<String> featureVectorizer,
                                                Preprocessor... preprocessors
                                               ) {
      return new PipelinedClassifier(classifier,
                                     featureVectorizer,
                                     preprocessors);
   }

   /**
    * Instantiates a new Pipelined classifier.
    *
    * @param classifier        the classifier
    * @param indexVectorizer   the index vectorizer
    * @param featureVectorizer the feature vectorizer
    * @param preprocessors     the preprocessors
    */
   public PipelinedClassifier(Classifier classifier,
                              Vectorizer<String> indexVectorizer,
                              Vectorizer<String> featureVectorizer,
                              Preprocessor... preprocessors
                             ) {
      super(indexVectorizer, featureVectorizer, new PreprocessorList(preprocessors));
      this.classifier = classifier;
   }

   /**
    * Instantiates a new Pipelined classifier.
    *
    * @param classifier        the classifier
    * @param featureVectorizer the feature vectorizer
    * @param preprocessors     the preprocessors
    */
   public PipelinedClassifier(Classifier classifier,
                              Vectorizer<String> featureVectorizer,
                              Preprocessor... preprocessors
                             ) {
      this(classifier, IndexVectorizer.labelVectorizer(), featureVectorizer, preprocessors);
   }

   @Override
   public Classifier copy() {
      PipelinedClassifier copy = new PipelinedClassifier(classifier.copy(), Cast.as(featureVectorizer));
      preprocessors.forEach(p -> copy.preprocessors.add(p.copy()));
      return copy;
   }

   @Override
   public NDArray estimate(NDArray data) {
      return classifier.estimate(data);
   }

   @Override
   public ClassifierEvaluation evaluate(Dataset evaluationData) {
      ClassifierEvaluation eval = getNumberOfLabels() == 2
                                  ? new BinaryEvaluation(getLabelVectorizer().decode(1.0).toString())
                                  : new MultiClassEvaluation(this);
      eval.evaluate(this, evaluationData);
      return eval;
   }

   @Override
   public void fit(SerializableSupplier<MStream<NDArray>> dataSupplier, FitParameters fitParameters) {
      classifier.fit(dataSupplier, fitParameters);
   }

   @Override
   public FitParameters getDefaultFitParameters() {
      return classifier.getDefaultFitParameters();
   }

   @Override
   public int getNumberOfFeatures() {
      return classifier.getNumberOfFeatures();
   }

   @Override
   public int getNumberOfLabels() {
      return classifier.getNumberOfLabels();
   }

   @Override
   public Classification predict(NDArray data) {
      return new Classification(estimate(data).getPredictedAsNDArray(), Cast.as(labelVectorizer));
   }

   /**
    * <p>Predicts the label(s) for a given example encoding and preprocessing the example as needed.</p>
    *
    * @param example the example
    * @return the classification result
    */
   public Classification predict(Example example) {
      return new Classification(estimate(example).getPredictedAsNDArray(), Cast.as(labelVectorizer));
   }
}//END OF PipelinedClassifier
