package com.gengoai.apollo.ml.clustering;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.linear.NDArrayFactory;
import com.gengoai.apollo.linear.NDArrayInitializer;
import com.gengoai.apollo.ml.FitParameters;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.apollo.ml.preprocess.Preprocessor;
import com.gengoai.apollo.ml.preprocess.PreprocessorList;
import com.gengoai.apollo.ml.vectorizer.Vectorizer;
import com.gengoai.apollo.stat.measure.Measure;
import com.gengoai.conversion.Cast;
import com.gengoai.function.SerializableSupplier;
import com.gengoai.stream.MStream;

import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Random;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * @author David B. Bracewell
 */
public class KMeans extends Clusterer {
   private static final long serialVersionUID = 1L;


   public KMeans(Preprocessor... preprocessors) {
      super(preprocessors);
   }

   public KMeans(Vectorizer<?> labelVectorizer, Vectorizer<String> featureVectorizer, PreprocessorList preprocessors) {
      super(labelVectorizer, featureVectorizer, preprocessors);
   }

   public KMeans(Vectorizer<?> labelVectorizer, Vectorizer<String> featureVectorizer, Preprocessor... preprocessors) {
      super(labelVectorizer, featureVectorizer, preprocessors);
   }

   public FlatClustering fit(SerializableSupplier<MStream<NDArray>> dataSupplier, Parameters fitParameters) {
      FlatClustering clusters = new FlatClustering();
      List<NDArray> instances = dataSupplier.get().collect();
      for (NDArray centroid : initCentroids(fitParameters.K, instances)) {
         Cluster c = new Cluster();
         c.setCentroid(centroid);
         clusters.add(c);
      }


      final Measure measure = fitParameters.measure;
      Map<NDArray, Integer> assignment = new ConcurrentHashMap<>();

      final AtomicLong numMoved = new AtomicLong(0);
      for (int itr = 0; itr < fitParameters.maxIterations; itr++) {
         clusters.forEach(Cluster::clear);
         numMoved.set(0);
         instances.parallelStream()
                  .forEach(ii -> {
                              int minI = 0;
                              double minD = measure.calculate(ii, clusters.get(0).getCentroid());
                              for (int ci = 1; ci < fitParameters.K; ci++) {
                                 double distance = measure.calculate(ii, clusters.get(ci).getCentroid());
                                 if (distance < minD) {
                                    minD = distance;
                                    minI = ci;
                                 }
                              }
                              Integer old = assignment.put(ii, minI);
                              clusters.get(minI).addPoint(ii);
                              if (old == null || old != minI) {
                                 numMoved.incrementAndGet();
                              }
                           }
                          );


         if (numMoved.get() == 0) {
            break;
         }


         for (int i = 0; i < fitParameters.K; i++) {
            clusters.get(i).getPoints().removeIf(Objects::isNull);
            if (clusters.get(i).size() == 0) {
               clusters.get(i).setCentroid(
                  NDArrayFactory.DEFAULT().create(NDArrayInitializer.rand(-1, 1), (int) instances.get(0).length()));
            } else {
               NDArray c = clusters.get(i).getCentroid().zero();
               for (NDArray ii : clusters.get(i)) {
                  if (ii != null) {
                     c.addi(ii);
                  }
               }
               c.divi((float) clusters.get(i).size());
            }
         }
      }

      for (int i = 0; i < clusters.size(); i++) {
         Cluster cluster = clusters.get(i);
         cluster.setId(i);
         if (cluster.size() > 0) {
            cluster.getPoints().removeIf(Objects::isNull);
            double average = cluster.getPoints().parallelStream()
                                    .flatMapToDouble(p1 -> cluster.getPoints()
                                                                  .stream()
                                                                  .filter(p2 -> p2 != p1)
                                                                  .mapToDouble(p2 -> measure.calculate(p1, p2)))
                                    .summaryStatistics()
                                    .getAverage();
            cluster.setScore(average);
         } else {
            cluster.setScore(Double.MAX_VALUE);
         }
      }
      return clusters;
   }


   private NDArray[] initCentroids(int K, List<NDArray> instances) {
      NDArray[] clusters = new NDArray[K];
      for (int i = 0; i < K; i++) {
         clusters[i] = NDArrayFactory.DEFAULT().zeros((int) instances.get(0).length());
      }
      double[] cnts = new double[K];
      Random rnd = new Random();
      for (NDArray ii : instances) {
         int ci = rnd.nextInt(K);
         clusters[ci].addi(ii);
         cnts[ci]++;
      }
      for (int i = 0; i < K; i++) {
         clusters[i].divi((float) cnts[i]);
      }
      return clusters;
   }


   @Override
   public Clustering fitPreprocessed(Dataset dataSupplier, FitParameters fitParameters) {
      return fit(() -> dataSupplier.stream().map(this::encode), Cast.as(fitParameters, Parameters.class));
   }


   @Override
   public Parameters getDefaultFitParameters() {
      return new Parameters();
   }


   /**
    * The type Parameters.
    */
   public static class Parameters extends Clusterer.ClusterParameters {
      /**
       * The K.
       */
      public int K = 2;
      /**
       * The Max iterations.
       */
      public int maxIterations = 100;

   }
}//END OF KMeans
