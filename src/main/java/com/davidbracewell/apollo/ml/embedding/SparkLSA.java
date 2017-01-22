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
import com.davidbracewell.apollo.linalg.DenseVector;
import com.davidbracewell.apollo.linalg.LabeledVector;
import com.davidbracewell.apollo.linalg.store.CosineSignature;
import com.davidbracewell.apollo.linalg.store.InMemoryLSH;
import com.davidbracewell.apollo.linalg.store.VectorStore;
import com.davidbracewell.apollo.ml.Encoder;
import com.davidbracewell.apollo.ml.EncoderPair;
import com.davidbracewell.apollo.ml.IndexEncoder;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.apollo.ml.sequence.Sequence;
import com.davidbracewell.collection.counter.Counter;
import com.davidbracewell.collection.counter.Counters;
import com.davidbracewell.stream.SparkStream;
import com.davidbracewell.stream.StreamingContext;
import lombok.Getter;
import lombok.Setter;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.SingularValueDecomposition;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.distributed.RowMatrix;
import scala.Tuple2;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * @author David B. Bracewell
 */
public class SparkLSA extends EmbeddingLearner {
   private static final long serialVersionUID = 1L;

   @Getter
   @Setter
   private int minCount = 5;
   @Getter
   @Setter
   private int maxVocab = 100_000;
   @Getter
   @Setter
   private double tolerance = 1e-10;
   @Getter
   @Setter
   private int dimension = 300;
   @Getter
   @Setter
   private double rCond = 1e-9;


   @Override
   protected Embedding trainImpl(Dataset<Sequence> dataset) {
      SparkStream<Counter<String>> stream = new SparkStream<>(dataset.stream().map(sequence -> {
         Counter<String> counter = Counters.newCounter();
         for (Instance instance : sequence) {
            counter.increment(
               instance.getFeatures().get(0).getName());
         }
         return counter.divideBySum();
      }));

      JavaPairRDD<String, Double> docFreqs = stream.getRDD()
                                                   .flatMap(c -> c.items().iterator())
                                                   .mapToPair(s -> new Tuple2<>(s, 1.0))
                                                   .reduceByKey(Math2::add)
                                                   .filter(t -> t._2() >= minCount);

      final Set<String> vocab = docFreqs.top(maxVocab,
                                             (Serializable & Comparator<Tuple2<String, Double>>) (t1, t2) -> -Double.compare(
                                                t1._2(), t2._2()))
                                        .stream()
                                        .map(Tuple2::_1)
                                        .collect(Collectors.toSet());

      final double N = dataset.size();
      final Map<String, Double> idf = docFreqs.filter(t -> vocab.contains(t._1())).mapToPair(
         t -> new Tuple2<>(t._1(), Math.log(N / t._2()))).collectAsMap();
      final int vocabSize = vocab.size();


      final Encoder featureEncoder = new IndexEncoder();
      featureEncoder.fit(StreamingContext.local().stream(vocab));

      JavaRDD<Vector> rowVectors = stream
                                      .getRDD()
                                      .map(c -> {
                                              Set<Tuple2<Integer, Double>> filtered =
                                                 c.filterByKey(vocab::contains)
                                                  .entries()
                                                  .stream()
                                                  .map(e -> new Tuple2<>((int) featureEncoder.encode(e.getKey()),
                                                                         idf.getOrDefault(e.getKey(), 1.0) * e.getValue()))
                                                  .collect(Collectors.toSet());
                                              return Vectors.sparse(vocabSize, filtered);
                                           }
                                          ).cache();

      RowMatrix mat = new RowMatrix(rowVectors.rdd());
      SingularValueDecomposition<RowMatrix, Matrix> svd = mat.computeSVD(dimension,
                                                                         true,
                                                                         rCond,
                                                                         Math.max(300, dimension * 3),
                                                                         tolerance,
                                                                         "auto");


      VectorStore<String> vectorStore = InMemoryLSH.builder()
                                                   .dimension(dimension)
                                                   .signatureSupplier(CosineSignature::new)
                                                   .createVectorStore();
      double[] v = svd.V().toArray();
      int offset = 0;
      for (int i = 0; i < vocabSize; i++) {
         vectorStore.add(new LabeledVector(featureEncoder.decode(i),
                                           new DenseVector(Arrays.copyOfRange(v, offset, offset + dimension))));
         offset += dimension;
      }

      return new Embedding(new EncoderPair(dataset.getLabelEncoder(), featureEncoder), vectorStore);
   }


   @Override
   public void reset() {

   }

}//END OF SparkLSA
