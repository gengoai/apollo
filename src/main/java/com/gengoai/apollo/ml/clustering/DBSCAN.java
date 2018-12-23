package com.gengoai.apollo.ml.clustering;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.ml.FitParameters;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.apollo.ml.preprocess.Preprocessor;
import com.gengoai.apollo.ml.preprocess.PreprocessorList;
import com.gengoai.apollo.ml.vectorizer.Vectorizer;
import com.gengoai.apollo.stat.measure.Distance;
import com.gengoai.apollo.stat.measure.DistanceMeasure;
import com.gengoai.conversion.Cast;
import com.gengoai.function.SerializableSupplier;
import com.gengoai.stream.MStream;
import org.apache.commons.math3.ml.clustering.DBSCANClusterer;

import java.util.List;

/**
 * @author David B. Bracewell
 */
public class DBSCAN extends FlatClusterer {

   public DBSCAN() {
   }

   public DBSCAN(Vectorizer<?> labelVectorizer, Vectorizer<String> featureVectorizer, PreprocessorList preprocessors) {
      super(labelVectorizer, featureVectorizer, preprocessors);
   }

   public DBSCAN(Vectorizer<String> featureVectorizer, PreprocessorList preprocessors) {
      super(featureVectorizer, preprocessors);
   }

   public DBSCAN(Vectorizer<?> labelVectorizer, Vectorizer<String> featureVectorizer, Preprocessor... preprocessors) {
      super(labelVectorizer, featureVectorizer, preprocessors);
   }

   public DBSCAN(Vectorizer<String> featureVectorizer, Preprocessor... preprocessors) {
      super(featureVectorizer, preprocessors);
   }

   public void fit(SerializableSupplier<MStream<NDArray>> dataSupplier, Parameters fitParameters) {
      DBSCANClusterer<ApacheClusterable> clusterer = new DBSCANClusterer<>(fitParameters.eps,
                                                                           fitParameters.minPts,
                                                                           new ApacheDistanceMeasure(
                                                                              fitParameters.distanceMeasure));
      this.measure = fitParameters.distanceMeasure;
      clusters.clear();
      List<ApacheClusterable> apacheClusterables = dataSupplier.get().map(ApacheClusterable::new).collect();

      List<org.apache.commons.math3.ml.clustering.Cluster<ApacheClusterable>> result = clusterer.cluster(
         apacheClusterables);

      for (int i = 0; i < result.size(); i++) {
         Cluster cp = new Cluster();
         cp.setId(i);
         cp.setCentroid(result.get(0).getPoints().get(0).getVector());
         clusters.add(cp);
      }

      apacheClusterables.forEach(a -> {
         NDArray n = a.getVector();
         int index = -1;
         double score = getMeasure().getOptimum().startingValue();
         for (int i = 0; i < clusters.size(); i++) {
            Cluster c = clusters.get(i);
            double s = getMeasure().calculate(n, c.getCentroid());
            if (getMeasure().getOptimum().test(s, score)) {
               index = i;
               score = s;
            }
         }
         clusters.get(index).addPoint(n);
      });

   }

   @Override
   public void fitPreprocessed(Dataset dataSupplier, FitParameters fitParameters) {
      fit(() -> dataSupplier.stream().map(this::encode), Cast.as(fitParameters, Parameters.class));
   }

   @Override
   public Parameters getDefaultFitParameters() {
      return new Parameters();
   }

   public static class Parameters extends FitParameters {
      /**
       * the maximum distance between two vectors to be in the same region
       */
      public double eps = 0.01;
      /**
       * the minimum number of points to form  a dense region
       */
      public int minPts = 2;
      /**
       * the distance measure to use for clustering
       */
      public DistanceMeasure distanceMeasure = Distance.Euclidean;
   }

}//END OF DBSCAN
