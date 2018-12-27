package com.gengoai.apollo.ml;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.apollo.ml.preprocess.Preprocessor;
import com.gengoai.apollo.ml.preprocess.PreprocessorList;
import com.gengoai.apollo.ml.vectorizer.Vectorizer;
import com.gengoai.conversion.Cast;
import com.gengoai.io.resource.Resource;

import java.io.Serializable;
import java.util.function.Consumer;

/**
 * <p>A generic interface to classification, regression, and sequence models. Models are trained using the
 * <code>fit</code> method which takes in the training data and {@link FitParameters}. Models are used on new data by
 * calling the <code>estimate</code> method.</p>
 * <p>Models can be trained and used to estimate directly on {@link NDArray}s or using {@link Dataset}s of {@link
 * Example}*s</p>
 *
 * @author David B. Bracewell
 */
public abstract class Model<T> implements Serializable {
   private static final long serialVersionUID = 1L;
   private final Vectorizer<String> featureVectorizer;
   private final Vectorizer<?> labelVectorizer;

   public PreprocessorList getPreprocessors() {
      return preprocessors;
   }

   private final PreprocessorList preprocessors;


   /**
    * Instantiates a new Model.
    *
    * @param labelVectorizer   the label vectorizer
    * @param featureVectorizer the feature vectorizer
    * @param preprocessors     the preprocessors
    */
   protected Model(Vectorizer<?> labelVectorizer,
                   Vectorizer<String> featureVectorizer,
                   Preprocessor... preprocessors
                  ) {
      this.featureVectorizer = featureVectorizer;
      this.labelVectorizer = labelVectorizer;
      this.preprocessors = new PreprocessorList(preprocessors);
   }

   /**
    * Instantiates a new Model.
    *
    * @param labelVectorizer   the label vectorizer
    * @param featureVectorizer the feature vectorizer
    * @param preprocessors     the preprocessors
    */
   protected Model(Vectorizer<?> labelVectorizer,
                   Vectorizer<String> featureVectorizer,
                   PreprocessorList preprocessors
                  ) {
      this.featureVectorizer = featureVectorizer;
      this.labelVectorizer = labelVectorizer;
      this.preprocessors = new PreprocessorList(preprocessors);
   }

   /**
    * Reads a Classifier from the given resource
    *
    * @param resource the resource containing the saved model
    * @return the deserialized (loaded) model
    * @throws Exception Something went wrong reading in the model
    */
   public static <T extends Model> T read(Resource resource) throws Exception {
      return resource.readObject();
   }

   /**
    * Convenience method for encoding and applying preprocessors to examples;
    *
    * @param example the example to encode and preprocess
    * @return the resulting encoded NDArray
    */
   public final NDArray encode(Example example) {
      NDArray array = featureVectorizer.transform(example);
      if (example.hasLabel()) {
         array.setLabel(labelVectorizer.transform(example));
      }
      return array;
   }


   /**
    * Preprocesses the example applying all preprocessors to it
    *
    * @param example the example
    * @return the example
    */
   public final Example preprocess(Example example) {
      return preprocessors.apply(example);
   }

   /**
    * Convenience method for encoding and applying preprocessors to examples;
    *
    * @param example the example to encode and preprocess
    * @return the resulting encoded NDArray
    */
   public final NDArray encodeAndPreprocess(Example example) {
      return encode(preprocess(example));
   }

   /**
    * Fits the model on the given {@link Dataset} using the model's default {@link FitParameters}.
    *
    * @param dataset the dataset to fit the model on
    */
   public final T fit(Dataset dataset) {
      return fit(dataset, getDefaultFitParameters());
   }

   /**
    * Fits the model on the given {@link Dataset} using the given consumer to modify the model's default {@link
    * FitParameters}**.
    *
    * @param dataset       the dataset
    * @param fitParameters the consumer to use to update the fit parameters
    */
   public final T fit(Dataset dataset, Consumer<? extends FitParameters> fitParameters) {
      FitParameters p = getDefaultFitParameters();
      fitParameters.accept(Cast.as(p));
      return fit(dataset, p);
   }

   /**
    * Fits the model on the given {@link Dataset} using the given {@link FitParameters}.
    *
    * @param dataset       the dataset
    * @param fitParameters the fit parameters
    */
   public final T fit(Dataset dataset, FitParameters fitParameters) {
      final Dataset preprocessed = preprocessors.fitAndTransform(dataset);
      labelVectorizer.fit(preprocessed);
      featureVectorizer.fit(preprocessed);
      return fitPreprocessed(preprocessed, fitParameters);
   }

   protected abstract T fitPreprocessed(Dataset preprocessed, FitParameters fitParameters);

   /**
    * Gets default fit parameters for the model.
    *
    * @return the default fit parameters
    */
   public abstract FitParameters getDefaultFitParameters();

   /**
    * Gets the feature vectorizer used by the model.
    *
    * @return the feature vectorizer
    */
   public Vectorizer<String> getFeatureVectorizer() {
      return featureVectorizer;
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
    * Gets the number of features in the model.
    *
    * @return the number of features
    */
   public int getNumberOfFeatures() {
      return featureVectorizer.size();
   }

   /**
    * Gets the number of labels in the model.
    *
    * @return the number of labels
    */
   public int getNumberOfLabels() {
      return labelVectorizer.size();
   }


   /**
    * Writes the model to the given resource.
    *
    * @param resource the resource to write the model to
    * @throws Exception Something went wrong writing the model
    */
   public void write(Resource resource) throws Exception {
      resource.setIsCompressed(true).writeObject(this);
   }
}//END OF Model
