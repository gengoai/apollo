package com.gengoai.apollo.ml.clustering;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.ml.FitParameters;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.apollo.ml.preprocess.Preprocessor;
import com.gengoai.apollo.ml.preprocess.PreprocessorList;
import com.gengoai.apollo.ml.vectorizer.IndexVectorizer;
import com.gengoai.apollo.ml.vectorizer.Vectorizer;
import com.gengoai.conversion.Cast;
import com.gengoai.function.SerializableSupplier;
import com.gengoai.stream.MStream;
import org.apache.commons.math3.ml.clustering.DBSCANClusterer;

import java.util.List;

/**
 * @author David B. Bracewell
 */
public class DBSCAN extends Clusterer {

   public DBSCAN(Preprocessor... preprocessors) {
      super(preprocessors);
   }

   public DBSCAN(Vectorizer<?> labelVectorizer, Vectorizer<String> featureVectorizer, PreprocessorList preprocessors) {
      super(labelVectorizer, featureVectorizer, preprocessors);
   }

   public DBSCAN(Vectorizer<?> labelVectorizer, Vectorizer<String> featureVectorizer, Preprocessor... preprocessors) {
      super(labelVectorizer, featureVectorizer, preprocessors);
   }

   public DBSCAN(Vectorizer<String> featureVectorizer, PreprocessorList preprocessors) {
      super(IndexVectorizer.labelVectorizer(), featureVectorizer, preprocessors);
   }

   public DBSCAN(Vectorizer<String> featureVectorizer, Preprocessor... preprocessors) {
      super(IndexVectorizer.labelVectorizer(), featureVectorizer, preprocessors);
   }

   public FlatClustering fit(SerializableSupplier<MStream<NDArray>> dataSupplier, Parameters fitParameters) {
      DBSCANClusterer<ApacheClusterable> clusterer = new DBSCANClusterer<>(fitParameters.eps,
                                                                           fitParameters.minPts,
                                                                           new ApacheDistanceMeasure(
                                                                              fitParameters.measure));
      FlatClustering centroids = new FlatClustering();
      List<ApacheClusterable> apacheClusterables = dataSupplier.get()
                                                               .parallel()
                                                               .map(ApacheClusterable::new)
                                                               .collect();

      List<org.apache.commons.math3.ml.clustering.Cluster<ApacheClusterable>> result = clusterer.cluster(
         apacheClusterables);
      for (int i = 0; i < result.size(); i++) {
         Cluster cp = new Cluster();
         cp.setId(i);
         cp.setCentroid(result.get(i).getPoints().get(0).getVector());
         centroids.add(cp);
      }

      apacheClusterables.forEach(a -> {
         NDArray n = a.getVector();
         int index = -1;
         double score = fitParameters.measure.getOptimum().startingValue();
         for (int i = 0; i < centroids.size(); i++) {
            Cluster c = centroids.get(i);
            double s = fitParameters.measure.calculate(n, c.getCentroid());
            if (fitParameters.measure.getOptimum().test(s, score)) {
               index = i;
               score = s;
            }
         }
         centroids.get(index).addPoint(n);
      });
      return centroids;
   }

   @Override
   public Clustering fitPreprocessed(Dataset dataSupplier, FitParameters fitParameters) {
      return fit(() -> dataSupplier.stream().map(this::encode), Cast.as(fitParameters, Parameters.class));
   }

   @Override
   public Parameters getDefaultFitParameters() {
      return new Parameters();
   }

   public static class Parameters extends Clusterer.ClusterParameters {
      /**
       * the maximum distance between two vectors to be in the same region
       */
      public double eps = 1.0;
      /**
       * the minimum number of points to form  a dense region
       */
      public int minPts = 2;
   }

}//END OF DBSCAN
