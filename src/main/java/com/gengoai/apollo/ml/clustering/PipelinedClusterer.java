package com.gengoai.apollo.ml.clustering;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.ml.FitParameters;
import com.gengoai.apollo.ml.PipelinedModel;
import com.gengoai.apollo.ml.preprocess.PreprocessorList;
import com.gengoai.apollo.ml.vectorizer.Vectorizer;
import com.gengoai.apollo.stat.measure.Measure;
import com.gengoai.function.SerializableSupplier;
import com.gengoai.stream.MStream;

import java.util.Iterator;
import java.util.List;
import java.util.stream.Collectors;

/**
 * @author David B. Bracewell
 */
public class PipelinedClusterer extends PipelinedModel implements Clusterer {
   private static final long serialVersionUID = 1L;
   private final Clusterer clusterer;

   /**
    * Instantiates a new Pipelined model.
    *
    * @param labelVectorizer   the label vectorizer
    * @param featureVectorizer the feature vectorizer
    */
   public PipelinedClusterer(Clusterer clusterer,
                             Vectorizer<?> labelVectorizer,
                             Vectorizer<?> featureVectorizer,
                             PreprocessorList preprocessors
                            ) {
      super(labelVectorizer, featureVectorizer, preprocessors);
      this.clusterer = clusterer;
   }

   public List<String> getLabels(int id) {
      return clusterer.getCluster(id)
                      .getPoints()
                      .stream()
                      .map(n -> getLabelVectorizer().decode(n.getLabelAsDouble()).toString())
                      .collect(Collectors.toList());
   }


   @Override
   public NDArray estimate(NDArray data) {
      return clusterer.estimate(data);
   }

   @Override
   public Cluster getCluster(int id) {
      return clusterer.getCluster(id);
   }

   @Override
   public Measure getMeasure() {
      return clusterer.getMeasure();
   }

   @Override
   public Cluster getRoot() {
      return clusterer.getRoot();
   }

   @Override
   public Iterator<Cluster> iterator() {
      return clusterer.iterator();
   }

   @Override
   public int size() {
      return clusterer.size();
   }


   @Override
   public void fit(SerializableSupplier<MStream<NDArray>> dataSupplier, FitParameters fitParameters) {
      clusterer.fit(dataSupplier, fitParameters);
   }

   @Override
   public FitParameters getDefaultFitParameters() {
      return clusterer.getDefaultFitParameters();
   }

   @Override
   public int getNumberOfFeatures() {
      return getFeatureVectorizer().size();
   }

   @Override
   public int getNumberOfLabels() {
      return 0;
   }

}//END OF PipelinedClusterer
