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
import com.davidbracewell.collection.Streams;
import com.davidbracewell.collection.counter.Counter;
import com.davidbracewell.conversion.Cast;
import com.davidbracewell.conversion.Val;
import com.davidbracewell.io.structured.ElementType;
import com.davidbracewell.io.structured.StructuredReader;
import com.davidbracewell.io.structured.StructuredWriter;
import com.davidbracewell.tuple.Tuple2;
import com.google.common.collect.Sets;
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
    * Convenience method for creating an instance from a collection of features.
    *
    * @param features the features
    * @param label    the label of the instance
    * @return the instance
    */
   public static Instance create(@NonNull Collection<Feature> features, Object label) {
      return new Instance(features, label);
   }

   /**
    * Convenience method for creating an instance from a vector. Feature names are string representations of the vector
    * indices.
    *
    * @param vector the vector
    * @return the instance
    */
   public static Instance fromVector(@NonNull com.davidbracewell.apollo.linalg.Vector vector) {
      List<Feature> features = Streams.asStream(vector.nonZeroIterator())
                                      .map(de -> Feature.real(Integer.toString(de.index), de.value))
                                      .collect(Collectors.toList());
      return create(features, vector.getLabel());
   }

   @Override
   public List<Instance> asInstances() {
      return Collections.singletonList(this);
   }

   @Override
   public Instance copy() {
      return new Instance(features.stream().map(Feature::copy).collect(Collectors.toList()), label);
   }

   @Override
   public Stream<String> getFeatureSpace() {
      return features.stream().map(Feature::getName);
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
      if (label == null) {
         return Stream.empty();
      }
      return Stream.of(label);
   }

   /**
    * Gets the value of the given feature.
    *
    * @param feature the feature name to look up
    * @return the value of the given feature or 0 if not in the instance
    */
   public double getValue(@NonNull String feature) {
      return features.stream().filter(f -> f.getName().equals(feature)).map(Feature::getValue).findFirst().orElse(0d);
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
                                     .map(f -> Feature.real(interner.intern(f.getName()), f.getValue()))
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

   @Override
   public void read(StructuredReader reader) throws IOException {
      this.label = null;
      this.features.clear();
      this.label = reader.nextKeyValue("label").cast();
      this.weight = reader.nextKeyValue("weight").asDoubleValue(1.0);
      reader.beginObject();
      while (reader.peek() != ElementType.END_OBJECT) {
         Tuple2<String, Val> fv = reader.nextKeyValue();
         this.features.add(Feature.real(fv.getKey(), fv.getValue().asDoubleValue()));
      }
      reader.endObject();
      this.features.trimToSize();
   }

   /**
    * Gets the features making up the instance as a stream
    *
    * @return the stream
    */
   public Stream<Feature> stream() {
      return features.stream();
   }

   /**
    * Converts the instance into a feature vector using the given encoder pair to map feature names and labels to double
    * values
    *
    * @param encoderPair the encoder pair
    * @return the vector
    */
   public FeatureVector toVector(@NonNull EncoderPair encoderPair) {
      FeatureVector vector = new FeatureVector(encoderPair);
      boolean isHash = encoderPair.getFeatureEncoder() instanceof HashingEncoder;
      features.forEach(f -> {
         int fi = (int) encoderPair.encodeFeature(f.getName());
         if (fi != -1) {
            if (isHash) {
               vector.set(fi, 1.0);
            } else {
               vector.set(fi, f.getValue());
            }
         }
      });
      vector.setLabel(encoderPair.encodeLabel(label));
      vector.setWeight(weight);
      return vector;
   }

   @Override
   public void write(@NonNull StructuredWriter writer) throws IOException {
      boolean inArray = writer.inArray();
      if (inArray) writer.beginObject();
      writer.writeKeyValue("label", label);
      writer.writeKeyValue("weight", weight);
      writer.beginObject("features");
      for (Feature f : features) {
         writer.writeKeyValue(f.getName(), f.getValue());
      }
      writer.endObject();
      if (inArray) writer.endObject();
   }

}//END OF Instance
