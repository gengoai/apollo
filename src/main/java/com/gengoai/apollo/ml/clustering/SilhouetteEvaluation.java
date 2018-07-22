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
 */

package com.gengoai.apollo.ml.clustering;

import com.gengoai.math.Math2;
import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.ml.Evaluation;
import com.gengoai.apollo.ml.Instance;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.apollo.stat.measure.Measure;
import com.gengoai.stream.StreamingContext;
import com.gengoai.string.TableFormatter;
import lombok.NonNull;

import java.io.PrintStream;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

import static com.gengoai.tuple.Tuples.$;

/**
 * @author David B. Bracewell
 */
public class SilhouetteEvaluation implements Evaluation<Instance, Clustering> {
   double avgSilhouette = 0;
   Map<Integer, Double> silhouette;

   @Override
   public void evaluate(@NonNull Clustering model, Dataset<Instance> dataset) {
      evaluate(model);
   }

   @Override
   public void evaluate(@NonNull Clustering model, Collection<Instance> dataset) {
      evaluate(model);
   }

   public void evaluate(@NonNull Clustering model) {
      Map<Integer, Cluster> idClusterMap = new HashMap<>();
      model.forEach(c -> idClusterMap.put(c.getId(), c));
      silhouette = StreamingContext.local().stream(idClusterMap.keySet())
                                   .parallel()
                                   .mapToPair(i -> $(i, silhouette(idClusterMap, i, model.getMeasure())))
                                   .collectAsMap();
      avgSilhouette = Math2.summaryStatistics(silhouette.values()).getAverage();
   }

   public double getAvgSilhouette() {
      return avgSilhouette;
   }

   public double getSilhoette(int id) {
      return silhouette.get(id);
   }

   @Override
   public void merge(@NonNull Evaluation<Instance, Clustering> evaluation) {
      throw new UnsupportedOperationException();
   }

   @Override
   public void output(@NonNull PrintStream printStream) {
      TableFormatter formatter = new TableFormatter();
      formatter.title("Silhouette Cluster Evaluation");
      formatter.header(Arrays.asList("Cluster", "Silhouette Score"));
      silhouette.keySet()
                .stream()
                .sorted()
                .forEach(id -> formatter.content(Arrays.asList(id, silhouette.get(id))));
      formatter.footer(Arrays.asList("Avg. Score", avgSilhouette));
      formatter.print(printStream);
   }

   public void reset() {
      this.avgSilhouette = 0;
      this.silhouette.clear();
   }

   public double silhouette(Map<Integer, Cluster> clusters, int index, Measure distanceMeasure) {
      Cluster c1 = clusters.get(index);
      if (c1.size() <= 1) {
         return 0;
      }

      double s = 0;
      for (NDArray point1 : c1) {
         double ai = 0;
         for (NDArray point2 : c1) {
            ai += distanceMeasure.calculate(point1, point2);
         }
         ai /= c1.size();
         double bi = clusters.keySet().parallelStream()
                             .filter(j -> j != index)
                             .mapToDouble(j -> {
                                double b = 0;
                                for (NDArray point2 : clusters.get(j)) {
                                   b += distanceMeasure.calculate(point1, point2);
                                }
                                return b;
                             }).min().orElseThrow(NullPointerException::new);
         s += (bi - ai) / Math.max(bi, ai);
      }

      return s / c1.size();
   }

}//END OF SilhouetteEvaluation
