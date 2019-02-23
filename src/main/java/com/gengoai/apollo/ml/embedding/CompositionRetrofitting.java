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

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.ml.vectorizer.DiscreteVectorizer;
import com.gengoai.io.resource.Resource;

/**
 * <p>
 * Retrofits a vector store through composing the vectors with the same term and optionally neighbors in a secondary
 * background vector store.
 * </p>
 *
 * @author David B. Bracewell
 */
public class CompositionRetrofitting implements Retrofitting {
   private static final long serialVersionUID = 1L;
   private final Embedding background;
   private int neighborSize = 0;
   private double neighborThreshold = 0.5;
   private double neighborWeight = 0;


   /**
    * Instantiates a new Composition retrofitting.
    *
    * @param background the background
    */
   public CompositionRetrofitting(Resource background) {
      try {
         this.background = Embedding.readWord2VecTextFormat(background);
      } catch (Exception e) {
         throw new RuntimeException(e);
      }
   }


   /**
    * Instantiates a new Composition retrofitting.
    *
    * @param background the background
    */
   public CompositionRetrofitting(Embedding background) {
      this.background = background;
   }


   @Override
   public Embedding apply(Embedding embedding) {
      Embedding out = new Embedding(embedding.getPipeline().copy());
      DiscreteVectorizer vectorizer = out.getPipeline().featureVectorizer;
      NDArray[] vectors = new NDArray[embedding.size()];
      for (String key : embedding.getAlphabet()) {
         int index = vectorizer.indexOf(key);
         NDArray tv = embedding.lookup(key).copy();
         if (background.contains(key)) {
            tv.addi(background.lookup(key));
            if (neighborSize > 0 && neighborWeight > 0) {
               background.query(VSQuery.termQuery(key)
                                       .limit(neighborSize)
                                       .threshold(neighborThreshold))
                         .forEach(n -> tv.addi(n.mul((float) neighborWeight)));
            }
         }
         vectors[index] = tv;
      }
      out.vectorIndex = new DefaultVectorIndex(vectors);
      return out;
   }


   /**
    * Gets neighbor size.
    *
    * @return the neighbor size
    */
   public int getNeighborSize() {
      return neighborSize;
   }

   /**
    * Sets neighbor size.
    *
    * @param neighborSize the neighbor size
    */
   public void setNeighborSize(int neighborSize) {
      this.neighborSize = neighborSize;
   }

   /**
    * Gets neighbor threshold.
    *
    * @return the neighbor threshold
    */
   public double getNeighborThreshold() {
      return neighborThreshold;
   }

   /**
    * Sets neighbor threshold.
    *
    * @param neighborThreshold the neighbor threshold
    */
   public void setNeighborThreshold(double neighborThreshold) {
      this.neighborThreshold = neighborThreshold;
   }

   /**
    * Gets neighbor weight.
    *
    * @return the neighbor weight
    */
   public double getNeighborWeight() {
      return neighborWeight;
   }

   /**
    * Sets neighbor weight.
    *
    * @param neighborWeight the neighbor weight
    */
   public void setNeighborWeight(double neighborWeight) {
      this.neighborWeight = neighborWeight;
   }
}// END OF CompositionRetrofitting
