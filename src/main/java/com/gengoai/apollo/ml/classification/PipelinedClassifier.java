package com.gengoai.apollo.ml.classification;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.ml.Dataset;
import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.FitParameters;
import com.gengoai.apollo.ml.vectorizer.IndexVectorizer;
import com.gengoai.apollo.ml.vectorizer.Vectorizer;
import com.gengoai.function.SerializableSupplier;
import com.gengoai.stream.MStream;

/**
 * @author David B. Bracewell
 */
public class PipelinedClassifier implements Classifier {
   private final Classifier classifier;
   private final Vectorizer<String> featureVectorizer;
   private final IndexVectorizer labelVectorizer = new IndexVectorizer(true);

   public PipelinedClassifier(Classifier classifier, Vectorizer<String> featureVectorizer) {
      this.classifier = classifier;
      this.featureVectorizer = featureVectorizer;
   }

   @Override
   public Classifier copy() {
      return new PipelinedClassifier(classifier.copy(), featureVectorizer);
   }

   @Override
   public NDArray estimate(NDArray data) {
      return classifier.estimate(data);
   }

   public void fit(Dataset dataset, FitParameters fitParameters) {
      labelVectorizer.fit(dataset);
      featureVectorizer.fit(dataset);
      classifier.fit(null, fitParameters);
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
   public Classification predict(NDArray data) {
      return new Classification(estimate(data), labelVectorizer);
   }

   public Classification predict(Example example) {
      return predict(featureVectorizer.transform(example));
   }
}//END OF PipelinedClassifier
