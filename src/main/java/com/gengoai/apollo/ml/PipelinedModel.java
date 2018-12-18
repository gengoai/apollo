package com.gengoai.apollo.ml;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.apollo.ml.preprocess.Preprocessor;
import com.gengoai.apollo.ml.preprocess.PreprocessorList;
import com.gengoai.apollo.ml.vectorizer.Vectorizer;
import com.gengoai.conversion.Cast;

import java.util.Collections;
import java.util.List;
import java.util.function.Consumer;

/**
 * <p>A {@link Model} that includes a feature and label vectorizer and preprocessors capable of working directly with
 * {@link Example} and {@link Dataset}s instead of {@link NDArray}.</p>
 *
 * @author David B. Bracewell
 */
public abstract class PipelinedModel implements Model {
   private static final long serialVersionUID = 1L;
   /**
    * The Feature vectorizer used to encode the features of an Example.
    */
   protected final Vectorizer<?> featureVectorizer;
   /**
    * The Label vectorizer used to encode the label of examples and decode the labels as needed.
    */
   protected final Vectorizer<?> labelVectorizer;
   /**
    * The Preprocessors used to transform the features and their values.
    */
   protected final PreprocessorList preprocessors;

   /**
    * Instantiates a new Pipelined model.
    *
    * @param labelVectorizer   the label vectorizer
    * @param featureVectorizer the feature vectorizer
    * @param preprocessors     the preprocessors
    */
   protected PipelinedModel(Vectorizer<?> labelVectorizer,
                            Vectorizer<?> featureVectorizer,
                            PreprocessorList preprocessors
                           ) {
      this.labelVectorizer = labelVectorizer;
      this.featureVectorizer = featureVectorizer;
      this.preprocessors = preprocessors;
   }


   /**
    * Convenience method for encoding and applying preprocessors to examples;
    *
    * @param example the example to encode and preprocess
    * @return the resulting encoded NDArray
    */
   protected final NDArray encodeAndPreprocess(Example example) {
      return encode(preprocessors.apply(example));
   }

   /**
    * Convenience method for encoding and applying preprocessors to examples;
    *
    * @param example the example to encode and preprocess
    * @return the resulting encoded NDArray
    */
   protected final NDArray encode(Example example) {
      NDArray array = featureVectorizer.transform(example);
      if (example.hasLabel()) {
         array.setLabel(labelVectorizer.transform(example));
      }
      return array;
   }

   /**
    * Use the model to make an estimate on the given example data.
    *
    * @param data the example to make an estimate for
    * @return the NDArray representing the estimate
    */
   public final NDArray estimate(Example data) {
      return estimate(encodeAndPreprocess(data));
   }

   /**
    * Evaluates the model on the given {@link Dataset}.
    *
    * @param evaluationData the evaluation data
    * @param evaluation     the evaluation
    * @return the evaluation
    */
   public Evaluation evaluate(Dataset evaluationData, Evaluation evaluation) {
      evaluationData.forEach(d -> evaluation.entry(estimate(d)));
      return evaluation;
   }

   /**
    * Fits the model on the given {@link Dataset} using the model's default {@link FitParameters}.
    *
    * @param dataset the dataset to fit the model on
    */
   public final void fit(Dataset dataset) {
      fit(dataset, getDefaultFitParameters());
   }

   /**
    * Fits the model on the given {@link Dataset} using the given consumer to modify the model's default {@link
    * FitParameters}**.
    *
    * @param dataset       the dataset
    * @param fitParameters the consumer to use to update the fit parameters
    */
   public final void fit(Dataset dataset, Consumer<? extends FitParameters> fitParameters) {
      FitParameters p = getDefaultFitParameters();
      fitParameters.accept(Cast.as(p));
      fit(dataset, p);
   }

   /**
    * Fits the model on the given {@link Dataset} using the given {@link FitParameters}.
    *
    * @param dataset       the dataset
    * @param fitParameters the fit parameters
    */
   public final void fit(Dataset dataset, FitParameters fitParameters) {
      labelVectorizer.fit(dataset);
      featureVectorizer.fit(dataset);
      final Dataset preprocessed = preprocessors.fitAndTransform(dataset);
      fitParameters.numLabels = labelVectorizer.size();
      fitParameters.numFeatures = featureVectorizer.size();
      fit(() -> preprocessed.stream().map(this::encode), fitParameters);
   }

   /**
    * Gets the feature vectorizer used by the model.
    *
    * @param <T> the type parameter
    * @return the feature vectorizer
    */
   public <T> Vectorizer<T> getFeatureVectorizer() {
      return Cast.as(featureVectorizer);
   }

   /**
    * Gets the label vectorizer used by the model.
    *
    * @param <T> the type parameter
    * @return the label vectorizer
    */
   public <T> Vectorizer<T> getLabelVectorizer() {
      return Cast.as(labelVectorizer);
   }


   /**
    * Gets an unmodifiable list of the preprocessors used by this model.
    *
    * @return the preprocessors
    */
   public List<Preprocessor> getPreprocessors() {
      return Collections.unmodifiableList(preprocessors);
   }
}//END OF PipelinedModel
