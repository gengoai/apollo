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

package com.davidbracewell.apollo.ml;

import com.davidbracewell.Interner;
import com.davidbracewell.apollo.linear.NDArray;
import com.davidbracewell.apollo.linear.NDArrayFactory;
import com.davidbracewell.apollo.ml.encoder.EncoderPair;
import com.davidbracewell.apollo.ml.encoder.HashingEncoder;
import com.davidbracewell.collection.counter.Counter;
import com.davidbracewell.conversion.Cast;
import com.davidbracewell.conversion.Val;
import com.davidbracewell.guava.common.collect.Sets;
import com.davidbracewell.json.JsonReader;
import com.davidbracewell.json.JsonTokenType;
import com.davidbracewell.json.JsonWriter;
import com.davidbracewell.tuple.Tuple2;
import lombok.*;

import java.io.IOException;
import java.io.Serializable;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * <p>A container for a set of features, associated label, and weight.</p>
 *
 * @author David B. Bracewell
 */
@EqualsAndHashCode
@ToString
public class Instance implements Example, Serializable, Iterable<Feature> {
   private static final long serialVersionUID = 1L;
   private final ArrayList<Feature> features;
   private Object label;
   @Getter
   @Setter
   private double weight = 1.0;

   /**
    * Instantiates a new Instance.
    *
    * @param features the features
    */
   public Instance(@NonNull Collection<Feature> features) {
      this(features, null);
   }

   /**
    * Instantiates a new Instance.
    *
    * @param features the features
    * @param label    the label
    */
   public Instance(@NonNull Collection<Feature> features, Object label) {
      this.features = new ArrayList<>(features);
      this.features.trimToSize();
      setLabel(label);
   }

   /**
    * Instantiates a new Instance.
    */
   public Instance() {
      this.features = new ArrayList<>();
      this.label = null;
   }

   /**
    * Convenience method for creating an instance from a collection of features.
    *
    * @param features the features
    * @return the instance
    */
   public static Instance create(@NonNull Collection<Feature> features) {
      return new Instance(features);
   }

   /**
    * Creates an instance from a counter containing feature name (keys) and their values.
    *
    * @param features the feature counter
    * @return the instance
    */
   public static Instance create(@NonNull Counter<String> features) {
      return create(features, null);
   }

   /**
    * Creates an instance from a counter containing feature name (keys) and their values.
    *
    * @param features the feature counter
    * @param label    the instance label
    * @return the instance
    */
   public static Instance create(@NonNull Counter<String> features, Object label) {
      List<Feature> featureList = features.entries()
                                          .stream()
                                          .map(e -> Feature.real(e.getKey(), e.getValue()))
                                          .collect(Collectors.toList());
      return create(featureList, label);
   }

   /**
    * Creates an instance from a map containing feature name (keys) and their values.
    *
    * @param features the feature map
    * @return the instance
    */
   public static Instance create(@NonNull Map<String, Double> features) {
      return create(features, null);
   }

   /**
    * Creates an instance from a map containing feature name (keys) and their values.
    *
    * @param features the feature map
    * @param label    the instance label
    * @return the instance
    */
   public static Instance create(@NonNull Map<String, Double> features, Object label) {
      List<Feature> featureList = features.entrySet()
                                          .stream()
                                          .map(e -> Feature.real(e.getKey(), e.getValue()))
                                          .collect(Collectors.toList());
      return create(featureList, label);
   }

   /**
    * Convenience method for creating an instance from a collection of features.
    *
    * @param features the features
    * @param label    the label of the instance
    * @return the instance
    */
   public static Instance create(@NonNull Collection<Feature> features, Object label) {
      return new Instance(features, label);
   }

//   /**
//    * Convenience method for creating an instance from a vector. Feature names are string representations of the vector
//    * indices.
//    *
//    * @param vector the vector
//    * @return the instance
//    */
//   public static Instance fromVector(@NonNull com.davidbracewell.apollo.linalg.Vector vector) {
//      List<Feature> features = Streams.asStream(vector.nonZeroIterator())
//                                      .map(de -> Feature.real(Integer.toString(de.index), de.value))
//                                      .collect(Collectors.toList());
//      return create(features, vector.getLabel());
//   }

   @Override
   public List<Instance> asInstances() {
      return Collections.singletonList(this);
   }

   @Override
   public Instance copy() {
      return new Instance(features.stream().map(Feature::copy).collect(Collectors.toList()), label);
   }

   @Override
   public void fromJson(JsonReader reader) throws IOException {
      this.label = null;
      this.features.clear();
      this.label = reader.nextKeyValue("label").cast();
      this.weight = reader.nextKeyValue("weight").asDoubleValue(1.0);
      reader.beginObject();
      while (reader.peek() != JsonTokenType.END_OBJECT) {
         Tuple2<String, Val> fv = reader.nextKeyValue();
         this.features.add(Feature.real(fv.getKey(), fv.getValue().asDoubleValue()));
      }
      reader.endObject();
      this.features.trimToSize();
   }

   @Override
   public Stream<String> getFeatureSpace() {
      return features.stream().map(Feature::getFeatureName);
   }

   /**
    * Gets the features of the instance.
    *
    * @return the features
    */
   public List<Feature> getFeatures() {
      return features;
   }

   /**
    * Gets the label of the instance.
    *
    * @return the label
    */
   public Object getLabel() {
      return label;
   }

   /**
    * Sets the label of the instance.
    *
    * @param label the label
    */
   public void setLabel(Object label) {
      if (label == null) {
         this.label = null;
      } else if (label instanceof Collection) {
         this.label = new HashSet<>(Cast.as(label));
      } else if (label instanceof Iterable) {
         this.label = Sets.newHashSet(Cast.<Iterable>as(label));
      } else if (label instanceof Iterator) {
         this.label = Sets.newHashSet(Cast.<Iterator>as(label));
      } else if (label.getClass().isArray()) {
         this.label = new HashSet<>(Arrays.asList(Cast.<Object[]>as(label)));
      } else {
         this.label = label;
      }
   }

   /**
    * Gets the label as a set. Useful for multilabel classification.
    *
    * @return the label set
    */
   public Set<Object> getLabelSet() {
      if (this.label == null) {
         return Collections.emptySet();
      } else if (this.label instanceof Set) {
         return Collections.unmodifiableSet(Cast.as(this.label));
      }
      return Collections.singleton(this.label);
   }

   @Override
   public Stream<Object> getLabelSpace() {
      return getLabelSet().stream();
   }

   /**
    * Gets the value of the given feature.
    *
    * @param feature the feature name to look up
    * @return the value of the given feature or 0 if not in the instance
    */
   public double getValue(@NonNull String feature) {
      return features.stream()
                     .filter(f -> f.getFeatureName().equals(feature))
                     .map(Feature::getValue)
                     .findFirst()
                     .orElse(0d);
   }

   /**
    * Determines if the instance has the given label.
    *
    * @param label the label to check
    * @return True if the instance has the given label, False otherwise
    */
   public boolean hasLabel(Object label) {
      return getLabelSet().contains(label);
   }

   /**
    * Determines if the instance has a label associated with it or not.
    *
    * @return True if the instance has a non-null label, False otherwise
    */
   public boolean hasLabel() {
      return label != null;
   }

   @Override
   public Instance intern(@NonNull Interner<String> interner) {
      return Instance.create(features.stream()
                                     .map(f -> Feature.real(interner.intern(f.getFeatureName()), f.getValue()))
                                     .collect(Collectors.toList()),
                             label
                            );
   }

   /**
    * Checks if this instance has a collection of labels.
    *
    * @return True if the instances has multiple labels, false if not
    */
   public boolean isMultiLabeled() {
      return this.label != null && (this.label instanceof Collection);
   }

   @Override
   public Iterator<Feature> iterator() {
      return this.features.iterator();
   }

   /**
    * Gets the features making up the instance as a stream
    *
    * @return the stream
    */
   public Stream<Feature> stream() {
      return features.stream();
   }

   @Override
   public void toJson(@NonNull JsonWriter writer) throws IOException {
      boolean inArray = writer.inArray();
      if (inArray) writer.beginObject();
      writer.property("label", label);
      writer.property("weight", weight);
      writer.beginObject("features");
      for (Feature f : features) {
         writer.property(f.getFeatureName(), f.getValue());
      }
      writer.endObject();
      if (inArray) writer.endObject();
   }

   /**
    * Converts the instance into a feature vector using the given encoder pair to map feature names and labels to double
    * values
    *
    * @param encoderPair the encoder pair
    * @param factory     The factory to use to create vectors
    * @return the vector
    */
   public <T> NDArray toVector(@NonNull EncoderPair encoderPair, @NonNull NDArrayFactory factory) {
      NDArray vector = factory.zeros(encoderPair.numberOfFeatures());
      boolean isBinary = encoderPair.getFeatureEncoder() instanceof HashingEncoder && Cast.<HashingEncoder>as(
         encoderPair.getFeatureEncoder()).isBinary();
      features.forEach(f -> {
         int fi = (int) encoderPair.encodeFeature(f.getFeatureName());
         if (fi != -1) {
            if (isBinary) {
               vector.increment(fi, 1.0);
            } else {
               vector.increment(fi, f.getValue());
            }
         }
      });
      if (label instanceof Iterable) {
         NDArray lblVector = factory.zeros(encoderPair.getLabelEncoder().size());
         for (Object lbl : Cast.<Iterable<Object>>as(label)) {
            lblVector.set((int) encoderPair.encodeLabel(lbl), 1.0);
         }
         vector.setLabel(lblVector);
      } else {
         vector.setLabel(encoderPair.encodeLabel(label));
      }
      vector.setWeight(weight);
      return vector;
   }

   /**
    * Converts the instance into a feature vector using the given encoder pair to map feature names and labels to double
    * values
    *
    * @param encoderPair the encoder pair
    * @return the vector
    */
   public NDArray toVector(@NonNull EncoderPair encoderPair) {
      return toVector(encoderPair, NDArrayFactory.SPARSE_DOUBLE);
   }

}//END OF Instance
