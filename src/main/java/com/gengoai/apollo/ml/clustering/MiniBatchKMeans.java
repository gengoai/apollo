/*
 * (c) 2005 David B. Bracewell
 *
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 *
 */

package com.gengoai.apollo.ml.clustering;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.ml.DiscretePipeline;
import com.gengoai.apollo.ml.FitParameters;
import com.gengoai.apollo.ml.Params;
import com.gengoai.apollo.ml.data.VectorizedDataset;
import com.gengoai.apollo.ml.preprocess.Preprocessor;
import com.gengoai.apollo.optimization.StoppingCriteria;
import com.gengoai.apollo.statistics.Sampling;
import com.gengoai.conversion.Cast;
import com.gengoai.logging.Logger;
import com.gengoai.tuple.Tuple2;
import com.gengoai.tuple.Tuple3;
import lombok.NonNull;

import java.util.List;
import java.util.PrimitiveIterator;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static com.gengoai.Validation.notNull;
import static com.gengoai.tuple.Tuples.$;

/**
 * <p>Implementation of KMeans using MiniBatch for better scalability.</p>
 *
 * @author David B. Bracewell
 */
public class MiniBatchKMeans extends FlatCentroidClusterer {
   private static final Logger log = Logger.getLogger(MiniBatchKMeans.class);

   /**
    * Instantiates a new Mini batch k means.
    *
    * @param preprocessors the preprocessors
    */
   public MiniBatchKMeans(Preprocessor... preprocessors) {
      super(preprocessors);
   }

   /**
    * Instantiates a new Mini batch k means.
    *
    * @param modelParameters the model parameters
    */
   public MiniBatchKMeans(DiscretePipeline modelParameters) {
      super(modelParameters);
   }


   private double iteration(VectorizedDataset stream, int[] counts, int batchSize) {
      //Select batch and compute the best cluster for each item
      List<Tuple3<NDArray, Integer, Double>> batch =
            Sampling.uniformInts(batchSize, 0, (int) stream.size(), true)
                    .parallel()
                    .mapToObj(i -> best(stream.get(i)).appendLeft(stream.get(i)))
                    .collect(Collectors.toList());

      //Update the centroids based on the assignments
      double diff = 0d;
      for(Tuple3<NDArray, Integer, Double> assignment : batch) {
         NDArray target = assignment.v1;
         int cid = assignment.v2;
         counts[cid]++;
         double eta = 1.0 / counts[cid];
         NDArray centroid = get(cid).getCentroid();
         centroid.muli(1.0 - eta).addi(target.mul(eta));
         diff += assignment.v3;
      }
      diff /= batch.size();
      return diff;
   }

   @Override
   public void fit(@NonNull VectorizedDataset vectors, @NonNull FitParameters<?> fitParameters) {
      Parameters p = notNull(Cast.as(fitParameters, Parameters.class));
      setMeasure(p.measure.value());

      PrimitiveIterator.OfInt itr = Sampling.uniformInts(p.K.value(), 0, (int) vectors.size(), false).iterator();
      for(int i = 0; i < p.K.value(); i++) {
         Cluster c = new Cluster();
         c.setId(i);
         c.setCentroid(vectors.get(itr.nextInt()).copy());
         add(c);
      }

      final int[] counts = new int[p.K.value()];
      StoppingCriteria.create("avg_distance", fitParameters)
                      .logger(log)
                      .untilTermination(iteration -> iteration(vectors, counts, p.batchSize.value()));

      //Assign examples to clusters
      final Integer[] locks = IntStream.range(0, p.K.value()).boxed().toArray(Integer[]::new);
      vectors.parallelStream()
             .forEach(v -> {
                Tuple2<Integer, Double> best = best(v);
                final Cluster c = get(best.v1);
                synchronized(locks[c.getId()]) {
                   c.addPoint(v);
                }
             });
   }

   private Tuple2<Integer, Double> best(NDArray v) {
      int bestId = 0;
      double bestMeasure = getMeasure().calculate(v, get(0).getCentroid());
      for(int j = 1; j < size(); j++) {
         double measure = getMeasure().calculate(v, get(j).getCentroid());
         if(getMeasure().getOptimum().test(measure, bestMeasure)) {
            bestId = j;
            bestMeasure = measure;
         }
      }
      return $(bestId, bestMeasure);
   }

   @Override
   public MiniBatchKMeans.Parameters getFitParameters() {
      return new Parameters();
   }

   /**
    * Fit Parameters for KMeans
    */
   public static class Parameters extends ClusterParameters<Parameters> {
      /**
       * The number of clusters
       */
      public final Parameter<Integer> K = parameter(Params.Clustering.K, 2);
      /**
       * The maximum number of iterations to run the clusterer for
       */
      public final Parameter<Integer> maxIterations = parameter(Params.Optimizable.maxIterations, 100);
      /**
       * The tolerance in change of in-group variance for determining if k-means has converged
       */
      public final Parameter<Double> tolerance = parameter(Params.Optimizable.tolerance, 1e-3);
      /**
       * The Batch size.
       */
      public final Parameter<Integer> batchSize = parameter(Params.Optimizable.batchSize, 1000);

      /**
       * The History size.
       */
      public final Parameter<Integer> historySize = parameter(Params.Optimizable.historySize, 10);
      /**
       * The Report interval.
       */
      public final Parameter<Integer> reportInterval = parameter(Params.Optimizable.reportInterval, 50);

   }
}//END OF MiniBatchKMeans
