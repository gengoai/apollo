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

package com.gengoai.apollo.ml;

import com.gengoai.Validation;
import com.gengoai.apollo.linear.NDArray;
import com.gengoai.string.Strings;

import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

import static com.gengoai.apollo.ml.Feature.booleanFeature;

/**
 * <p>An Instance represents an example over a single object. It has a {@link #size()} of <code>1</code> and is not
 * allowed to add any child examples. Instances are used for input to classification and regression problems and as the
 * child examples for sequence labeling problems.</p>
 *
 * @author David B. Bracewell
 */
public class Instance extends Example {
   private static final long serialVersionUID = 1L;
   private final List<Feature> features;


   /**
    * Instantiates a new Instance with a null label and no features defined.
    */
   public Instance() {
      this.features = new ArrayList<>();
   }

   /**
    * Instantiates a new Instance with the given label and features.
    *
    * @param label    the label
    * @param features the features
    */
   public Instance(Object label, Feature... features) {
      this(label, Arrays.asList(features));
   }

   /**
    * Instantiates a new Instance with  the given label and features.
    *
    * @param label    the label
    * @param features the features
    */
   public Instance(Object label, List<Feature> features) {
      this.features = new ArrayList<>(features);
      setLabel(label);
   }

   /**
    * Creates a special Instance denoting the begin of sequence (or sentence), which has a label and single True feature
    * named <code>__BOS-INDEX__</code>, where <code>INDEX</code> is the offset from the <code>0</code> index. This
    * instance will return the single feature name with given prefix on all calls to {@link
    * #getFeatureByPrefix(String)}.
    *
    * @param distanceFromBegin the offset from the beginning of the sequence (i.e. index 0, e.g. -1)
    * @return the special beginning of sequence example at the given offset
    */
   public static Example BEGIN_OF_SEQUENCE(int distanceFromBegin) {
      String name = "__BOS-" + Math.abs(distanceFromBegin) + "__";
      return new Instance(name, booleanFeature(name)) {
         @Override
         public Feature getFeatureByPrefix(String prefix) {
            return booleanFeature(Strings.appendIfNotPresent(prefix, "=") + name);
         }
      };
   }

   /**
    * Creates a special Instance denoting the end of sequence (or sentence), which has a label and single True feature
    * named <code>__EOS-INDEX+1__</code>, where <code>INDEX</code> is the offset from the size of the example in the
    * sequence. This instance will return the single feature name with given prefix on all calls to {@link
    * #getFeatureByPrefix(String)}.
    *
    * @param distanceFromEnd the offset from the size of the example. (e.g. if the size is 4 an offset could be 0 when
    *                        the index is at 4, 1 when the index is at 5, etc.)
    * @return the special end of sequence example at the given offset
    */
   public static Example END_OF_SEQUENCE(int distanceFromEnd) {
      String name = "__EOS-" + (distanceFromEnd + 1) + "__";
      return new Instance(name, booleanFeature(name)) {
         @Override
         public Feature getFeatureByPrefix(String prefix) {
            return booleanFeature(Strings.appendIfNotPresent(prefix, "=") + name);
         }
      };
   }

   @Override
   public Example copy() {
      Instance copy = new Instance(getLabel(), features);
      copy.setWeight(getWeight());
      return copy;
   }

   @Override
   public boolean equals(Object o) {
      if (this == o) return true;
      if (!(o instanceof Instance)) return false;
      Instance instance = (Instance) o;
      return Objects.equals(features, instance.features) &&
                Objects.equals(getLabel(), instance.getLabel());
   }

   @Override
   public Example getExample(int index) {
      Validation.checkPositionIndex(index, 1);
      return this;
   }

   @Override
   public Feature getFeatureByPrefix(String prefix, Feature defaultValue) {
      return features.stream()
                     .filter(f -> f.hasPrefix(prefix))
                     .findFirst()
                     .orElse(defaultValue);
   }

   @Override
   public List<Feature> getFeatures() {
      return features;
   }


   @Override
   public int hashCode() {
      return Objects.hash(features, getLabel());
   }

   @Override
   public boolean isInstance() {
      return true;
   }

   @Override
   public Instance mapFeatures(Function<? super Feature, Optional<Feature>> mapper) {
      Instance ii = new Instance(getLabel(), features.stream()
                                                     .map(mapper)
                                                     .filter(Optional::isPresent)
                                                     .map(Optional::get)
                                                     .collect(Collectors.toList()));
      ii.setWeight(getWeight());
      return ii;
   }

   @Override
   public Example mapInstance(Function<Instance, Instance> mapper) {
      return mapper.apply(this);
   }

   @Override
   public int size() {
      return 1;
   }

   @Override
   public String toString() {
      return "Instance{" +
                "features=" + features +
                ", label=" + getLabel() +
                ", weight=" + getWeight() +
                '}';
   }

   @Override
   public NDArray transform(Pipeline pipeline) {
      NDArray array = pipeline.featureVectorizer.transform(this);
      if (hasLabel()) {
         array.setLabel(pipeline.labelVectorizer.transform(this));
      }
      return array.setWeight(getWeight()).compact();
   }
}//END OF Instance
