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

package com.gengoai.apollo.ml.preprocess;

import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.Feature;
import com.gengoai.apollo.ml.Instance;
import com.gengoai.apollo.ml.data.ExampleDataset;
import com.gengoai.collection.multimap.ArrayListMultimap;
import com.gengoai.collection.multimap.Multimap;
import com.gengoai.math.Math2;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

/**
 * <p>Collapses sequences into a single instance.</p>
 *
 * @author David B. Bracewell
 */
public class Sequence2Instance implements Preprocessor, Serializable {
   private static final long serialVersionUID = 1L;

   /**
    * How to combine the sequence items.
    */
   public enum Mode {
      /**
       * Averages the values of the features
       */
      Average {
         @Override
         public double combine(Collection<Double> values) {
            return Math2.sum(values) / values.size();
         }
      },
      /**
       * Use the maximum value of the feature
       */
      Max {
         @Override
         public double combine(Collection<Double> values) {
            return Collections.max(values);
         }
      },
      /**
       * Use the minimum value of the feature
       */
      Min {
         @Override
         public double combine(Collection<Double> values) {
            return Collections.min(values);
         }
      },
      /**
       * Sums the values
       */
      Sum {
         @Override
         public double combine(Collection<Double> values) {
            return Math2.sum(values);
         }
      },
      /**
       * Treats all features as binary
       */
      Binary {
         @Override
         public double combine(Collection<Double> values) {
            return 1;
         }
      };


      /**
       * Methodology for combining the values of a feature across one or more instances.
       *
       * @param values the values
       * @return the combined new value
       */
      public abstract double combine(Collection<Double> values);
   }

   private Mode mode;

   /**
    * Instantiates a new Sequence2Instance.
    *
    * @param mode How to combine the values
    */
   public Sequence2Instance(Mode mode) {
      this.mode = mode;
   }


   @Override
   public Example apply(Example example) {
      if (example.isInstance()) {
         return example;
      }
      Multimap<String, Double> features = new ArrayListMultimap<>();
      List<String> labels = new ArrayList<>();
      example.forEach(e -> {
         for (Feature feature : e.getFeatures()) {
            features.put(feature.getName(), feature.getValue());
         }
         labels.add(e.getLabel());
      });
      return new Instance(labels, features.keySet().stream()
                                          .map(name -> Feature.realFeature(name, mode.combine(features.get(name))))
                                          .collect(Collectors.toList()));
   }

   @Override
   public ExampleDataset fitAndTransform(ExampleDataset dataset) {
      return dataset.map(this::apply);
   }

   @Override
   public void reset() {

   }
}//END OF Sequence2Instance
