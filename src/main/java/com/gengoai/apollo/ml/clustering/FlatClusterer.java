package com.gengoai.apollo.ml.clustering;

import com.gengoai.apollo.ml.preprocess.Preprocessor;
import com.gengoai.apollo.ml.preprocess.PreprocessorList;
import com.gengoai.apollo.ml.vectorizer.IndexVectorizer;
import com.gengoai.apollo.ml.vectorizer.Vectorizer;
import com.gengoai.apollo.stat.measure.Measure;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * @author David B. Bracewell
 */
public abstract class FlatClusterer extends Clusterer {
   private static final long serialVersionUID = 1L;
   protected final List<Cluster> clusters = new ArrayList<>();
   protected Measure measure;

   public FlatClusterer(Preprocessor... preprocessors) {
      super(IndexVectorizer.labelVectorizer(),
            IndexVectorizer.featureVectorizer(),
            preprocessors);
   }

   public FlatClusterer(Vectorizer<?> labelVectorizer, Vectorizer<String> featureVectorizer, PreprocessorList preprocessors) {
      super(labelVectorizer, featureVectorizer, preprocessors);
   }

   public FlatClusterer(Vectorizer<String> featureVectorizer, PreprocessorList preprocessors) {
      super(IndexVectorizer.labelVectorizer(), featureVectorizer, preprocessors);
   }

   public FlatClusterer(Vectorizer<?> labelVectorizer, Vectorizer<String> featureVectorizer, Preprocessor... preprocessors) {
      super(labelVectorizer, featureVectorizer, preprocessors);
   }

   public FlatClusterer(Vectorizer<String> featureVectorizer, Preprocessor... preprocessors) {
      super(IndexVectorizer.labelVectorizer(), featureVectorizer, preprocessors);
   }

   @Override
   public Measure getMeasure() {
      return measure;
   }

   @Override
   public Cluster getCluster(int index) {
      return clusters.get(index);
   }

   @Override
   public Cluster getRoot() {
      throw new UnsupportedOperationException();
   }

   @Override
   public boolean isFlat() {
      return true;
   }

   @Override
   public Iterator<Cluster> iterator() {
      return clusters.iterator();
   }

   @Override
   public int size() {
      return clusters.size();
   }

}//END OF FlatClusterer
