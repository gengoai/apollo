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

package com.davidbracewell.apollo.ml.embedding;

import com.davidbracewell.Math2;
import com.davidbracewell.apollo.affinity.AssociationMeasures;
import com.davidbracewell.apollo.affinity.ContingencyTable;
import com.davidbracewell.apollo.affinity.ContingencyTableCalculator;
import com.davidbracewell.apollo.linalg.LabeledVector;
import com.davidbracewell.apollo.linalg.SparkLinearAlgebra;
import com.davidbracewell.apollo.linalg.store.CosineSignature;
import com.davidbracewell.apollo.linalg.store.InMemoryLSH;
import com.davidbracewell.apollo.linalg.store.VectorStore;
import com.davidbracewell.apollo.ml.Encoder;
import com.davidbracewell.apollo.ml.EncoderPair;
import com.davidbracewell.apollo.ml.IndexEncoder;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.apollo.ml.sequence.Sequence;
import com.davidbracewell.collection.counter.MultiCounter;
import com.davidbracewell.stream.StreamingContext;
import com.davidbracewell.stream.accumulator.MMultiCounterAccumulator;
import lombok.Getter;
import lombok.Setter;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.distributed.RowMatrix;

import java.util.Map;

import static com.davidbracewell.tuple.Tuples.$;

/**
 * @author David B. Bracewell
 */
public class SVDEmbedding extends EmbeddingLearner {
   private static final long serialVersionUID = 1L;
   @Getter
   @Setter
   private int dimension = 300;
   @Getter
   @Setter
   private int windowSize;
   @Getter
   @Setter
   private ContingencyTableCalculator calculator = AssociationMeasures.PPMI;

   @Override
   public void reset() {

   }

   @Override
   protected Embedding trainImpl(Dataset<Sequence> dataset) {

      final Map<String, Double> unigrams = dataset
                                              .stream()
                                              .flatMap(sequence -> sequence
                                                                      .asInstances()
                                                                      .stream()
                                                                      .flatMap(Instance::getFeatureSpace)
                                                      )
                                              .mapToPair(s -> $(s, 1.0))
                                              .reduceByKey(Math2::add)
                                              .collectAsMap();

      final Encoder featureEncoder = new IndexEncoder();
      featureEncoder.fit(StreamingContext.local().stream(unigrams.keySet()));

      MMultiCounterAccumulator<Integer, Integer> accumulator = dataset.getStreamingContext().multiCounterAccumulator();
      dataset.stream()
             .forEach(sequence -> {
                for (int i = 0; i < sequence.size(); i++) {
                   if (sequence.get(i).getFeatures().size() > 0) {
                      int iFeature = (int) featureEncoder.encode(sequence.get(i).getFeatures().get(0).getName());
                      for (int j = Math.min(i - windowSize, 0); j <= Math.max(sequence.size()-1, i + windowSize); j++) {
                         if (sequence.get(j).getFeatures().size() > 0) {
                            int jFeature = (int) featureEncoder.encode(sequence.get(j)
                                                                               .getFeatures()
                                                                               .get(0)
                                                                               .getName());
                            accumulator.increment(Math.min(iFeature, jFeature), Math.max(iFeature, jFeature));
                         }
                      }
                   }
                }
             });


      MultiCounter<Integer, Integer> windowCounts = accumulator.value();
      final double totalCounts = unigrams.values().parallelStream().count();
      RowMatrix mat = new RowMatrix(
                                      StreamingContext.distributed().range(0, featureEncoder.size())
                                                      .map(i -> {
                                                         double[] v = new double[featureEncoder.size()];
                                                         double iCount = unigrams.get(featureEncoder.decode(i).toString());
                                                         for (int j = 0; j < featureEncoder.size(); j++) {
                                                            double jCount = unigrams.get(
                                                               featureEncoder.decode(j).toString());
                                                            double n11 = Math.max(windowCounts.get(i, j),
                                                                            windowCounts.get(j, i));

                                                            v[j] = calculator.calculate(
                                                               ContingencyTable.create2X2(n11,iCount, jCount, totalCounts));
                                                         }
                                                         return (Vector) new DenseVector(v);
                                                      })
                                                      .getRDD()
                                                      .cache()
                                                      .rdd());


//        SparkStream<Counter<String>> stream = new SparkStream<>(dataset
//                                                                    .stream()
//                                                                    .map(sequence -> {
//                                                                        Counter<String> counter = Counters.newCounter();
//                                                                        for (Instance instance : sequence) {
//                                                                            counter.increment(
//                                                                                instance
//                                                                                    .getFeatures()
//                                                                                    .get(0)
//                                                                                    .getName());
//                                                                        }
//                                                                        return counter.divideBySum();
//                                                                    }));
//
//        JavaPairRDD<String, Double> docFreqs = stream
//                                                   .getRDD()
//                                                   .flatMap(c -> c
//                                                                     .items()
//                                                                     .iterator())
//                                                   .mapToPair(s -> new Tuple2<>(s, 1.0))
//                                                   .reduceByKey(Math2::add)
//
//        final Set<String> vocab = docFreqs
//                                      .top(maxVocab,
//                                           (Serializable & Comparator<Tuple2<String, Double>>) (t1, t2) -> -Double.compare(
//                                               t1._2(), t2._2()))
//                                      .stream()
//                                      .map(Tuple2::_1)
//                                      .collect(Collectors.toSet());
//
//        final double N = dataset.size();
//        final Map<String, Double> idf = docFreqs
//                                            .filter(t -> vocab.contains(t._1()))
//                                            .mapToPair(
//                                                t -> new Tuple2<>(t._1(), Math.log(N / t._2())))
//                                            .collectAsMap();
//        final int vocabSize = vocab.size();
//
//
//        final Encoder featureEncoder = new IndexEncoder();
//        featureEncoder.fit(StreamingContext
//                               .local()
//                               .stream(vocab));
//
//        JavaRDD<Vector> rowVectors = stream
//                                         .getRDD()
//                                         .map(c -> {
//                                                  Set<Tuple2<Integer, Double>> filtered =
//                                                      c
//                                                          .filterByKey(vocab::contains)
//                                                          .entries()
//                                                          .stream()
//                                                          .map(e -> new Tuple2<>((int) featureEncoder.encode(e.getKey()),
//                                                                                 idf.getOrDefault(e.getKey(),
//                                                                                                  1.0) * e.getValue()))
//                                                          .collect(Collectors.toSet());
//                                                  return Vectors.sparse(vocabSize, filtered);
//                                              }
//                                         )
//                                         .cache();
//
//        RowMatrix mat = new RowMatrix(rowVectors.rdd());
//
      VectorStore<String> vectorStore = InMemoryLSH
                                           .builder()
                                           .dimension(dimension)
                                           .signatureSupplier(CosineSignature::new)
                                           .createVectorStore();

      SparkLinearAlgebra
         .sparkSVD(mat, dimension)
         .U()
         .rows()
         .toJavaRDD()
         .map(Vector::toArray)
         .zipWithIndex()
         .toLocalIterator()
         .forEachRemaining(t -> vectorStore.add(new LabeledVector(featureEncoder.decode(t._2().intValue()),
                                                                  new com.davidbracewell.apollo.linalg.DenseVector(t._1()))));
      return new Embedding(new EncoderPair(dataset.getLabelEncoder(), featureEncoder), vectorStore);
   }

}//END OF SparkLSA
