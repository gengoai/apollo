package com.gengoai.apollo.ml.clustering;

import com.gengoai.apollo.ml.FitParameters;
import com.gengoai.apollo.ml.Model;
import com.gengoai.apollo.ml.preprocess.Preprocessor;
import com.gengoai.apollo.ml.preprocess.PreprocessorList;
import com.gengoai.apollo.ml.vectorizer.IndexVectorizer;
import com.gengoai.apollo.ml.vectorizer.Vectorizer;
import com.gengoai.apollo.stat.measure.Distance;
import com.gengoai.apollo.stat.measure.Measure;

/**
 * The type Clusterer.
 *
 * @author David B. Bracewell
 */
public abstract class Clusterer extends Model<Clustering> {
   private static final long serialVersionUID = 1L;

   /**
    * Instantiates a new Clusterer.
    *
    * @param preprocessors the preprocessors
    */
   public Clusterer(Preprocessor... preprocessors) {
      super(IndexVectorizer.labelVectorizer(),
            IndexVectorizer.featureVectorizer(),
            preprocessors);
   }

   /**
    * Instantiates a new Clusterer.
    *
    * @param labelVectorizer   the label vectorizer
    * @param featureVectorizer the feature vectorizer
    * @param preprocessors     the preprocessors
    */
   public Clusterer(Vectorizer<?> labelVectorizer, Vectorizer<String> featureVectorizer, PreprocessorList preprocessors) {
      super(labelVectorizer, featureVectorizer, preprocessors);
   }


   /**
    * Instantiates a new Clusterer.
    *
    * @param labelVectorizer   the label vectorizer
    * @param featureVectorizer the feature vectorizer
    * @param preprocessors     the preprocessors
    */
   public Clusterer(Vectorizer<?> labelVectorizer, Vectorizer<String> featureVectorizer, Preprocessor... preprocessors) {
      super(labelVectorizer, featureVectorizer, preprocessors);
   }


   public static class ClusterParameters extends FitParameters {
      public Measure measure = Distance.Euclidean;

   }

}//END OF Clusterer
