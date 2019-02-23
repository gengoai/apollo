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

package com.gengoai.apollo.ml.embedding;

/**
 * <p>
 * Retrofits a vector store through composing the vectors with the same term and optionally neighbors in a secondary
 * background vector store.
 * </p>
 *
 * @author David B. Bracewell
 */
public class CompositionRetrofitting {
//
//} implements Retrofitting {
//   private static final long serialVersionUID = 1L;
//   private final VectorStore background;
//   private int neighborSize = 0;
//   private double neighborThreshold = 0.5;
//   private double neighborWeight = 0;
//
//
//   /**
//    * Instantiates a new Composition retrofitting.
//    *
//    * @param background the background
//    */
//   public CompositionRetrofitting(Resource background) {
//      try {
//         this.background = VectorStore.read(background);
//      } catch (Exception e) {
//         throw new RuntimeException(e);
//      }
//   }
//
//
//   /**
//    * Instantiates a new Composition retrofitting.
//    *
//    * @param background the background
//    */
//   public CompositionRetrofitting(VectorStore background) {
//      this.background = background;
//   }
//
//
//   @Override
//   public VectorStore apply(VectorStore embedding) {
//      VSBuilder newEmbedding = VectorStore.builder(embedding.getParameters());
//      embedding.keySet().forEach(term -> {
//         NDArray tv = embedding.get(term).copy();
//         if (background.containsKey(term)) {
//            tv.addi(background.get(term));
//            if (neighborSize > 0 && neighborWeight > 0) {
//               background.query(VSQuery.termQuery(term)
//                                       .limit(neighborSize)
//                                       .threshold(neighborThreshold))
//                         .forEach(n -> tv.addi(n.mul((float) neighborWeight)));
//            }
//         }
//         newEmbedding.add(term, tv);
//      });
//      return newEmbedding.build();
//   }
//
//
//   /**
//    * Gets neighbor size.
//    *
//    * @return the neighbor size
//    */
//   public int getNeighborSize() {
//      return neighborSize;
//   }
//
//   /**
//    * Sets neighbor size.
//    *
//    * @param neighborSize the neighbor size
//    */
//   public void setNeighborSize(int neighborSize) {
//      this.neighborSize = neighborSize;
//   }
//
//   /**
//    * Gets neighbor threshold.
//    *
//    * @return the neighbor threshold
//    */
//   public double getNeighborThreshold() {
//      return neighborThreshold;
//   }
//
//   /**
//    * Sets neighbor threshold.
//    *
//    * @param neighborThreshold the neighbor threshold
//    */
//   public void setNeighborThreshold(double neighborThreshold) {
//      this.neighborThreshold = neighborThreshold;
//   }
//
//   /**
//    * Gets neighbor weight.
//    *
//    * @return the neighbor weight
//    */
//   public double getNeighborWeight() {
//      return neighborWeight;
//   }
//
//   /**
//    * Sets neighbor weight.
//    *
//    * @param neighborWeight the neighbor weight
//    */
//   public void setNeighborWeight(double neighborWeight) {
//      this.neighborWeight = neighborWeight;
//   }
}// END OF CompositionRetrofitting
