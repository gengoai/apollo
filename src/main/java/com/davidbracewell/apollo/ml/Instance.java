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

import com.davidbracewell.collection.Collect;
import com.davidbracewell.collection.Interner;
import com.davidbracewell.conversion.Cast;
import com.davidbracewell.conversion.Val;
import com.davidbracewell.io.structured.ElementType;
import com.davidbracewell.io.structured.StructuredReader;
import com.davidbracewell.io.structured.StructuredWriter;
import com.davidbracewell.tuple.Tuple2;
import com.google.common.collect.Sets;
import lombok.EqualsAndHashCode;
import lombok.NonNull;
import lombok.ToString;

import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * The type Instance.
 *
 * @author David B. Bracewell
 */
@EqualsAndHashCode
@ToString
public class Instance implements Example, Serializable, Iterable<Feature> {
  private static final long serialVersionUID = 1L;
  private final ArrayList<Feature> features;
  private Object label;

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
   * Create instance.
   *
   * @param features the features
   * @return the instance
   */
  public static Instance create(@NonNull Collection<Feature> features) {
    return new Instance(features);
  }

  /**
   * Create instance.
   *
   * @param features the features
   * @param label    the label
   * @return the instance
   */
  public static Instance create(@NonNull Collection<Feature> features, Object label) {
    return new Instance(features, label);
  }


  /**
   * Gets features by prefix.
   *
   * @param prefix the prefix
   * @return the features by prefix
   */
  public List<Feature> getFeaturesByPrefix(@NonNull String prefix) {
    return features.stream().filter(f -> f.getName().startsWith(prefix)).collect(Collectors.toList());
  }

  public Optional<Feature> getFeature(@NonNull String name) {
    return features.stream().filter(f -> f.getName().equals(name)).findFirst();
  }

  public Optional<Feature> getFeatureByPrefix(@NonNull String name) {
    return features.stream().filter(f -> f.getName().startsWith(name)).findFirst();
  }

  /**
   * Gets value.
   *
   * @param feature the feature
   * @return the value
   */
  public double getValue(@NonNull String feature) {
    return features.stream().filter(f -> f.getName().equals(feature)).map(Feature::getValue).findFirst().orElse(0d);
  }

  /**
   * Gets label.
   *
   * @return the label
   */
  public Object getLabel() {
    return label;
  }

  /**
   * Sets label.
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
   * Is multi labeled boolean.
   *
   * @return the boolean
   */
  public boolean isMultiLabeled() {
    return this.label != null && (this.label instanceof Collection);
  }

  /**
   * Gets label set.
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


  /**
   * Has label boolean.
   *
   * @param label the label
   * @return the boolean
   */
  public boolean hasLabel(Object label) {
    return getLabelSet().contains(label);
  }

  /**
   * Has label boolean.
   *
   * @return the boolean
   */
  public boolean hasLabel() {
    return label != null;
  }

  @Override
  public Iterator<Feature> iterator() {
    return this.features.iterator();
  }

  @Override
  public Instance copy() {
    return new Instance(features.stream().map(Feature::copy).collect(Collectors.toList()), label);
  }

  @Override
  public Stream<String> getFeatureSpace() {
    return features.stream().map(Feature::getName).distinct();
  }

  /**
   * Stream stream.
   *
   * @return the stream
   */
  public Stream<Feature> stream() {
    return Collect.from(this);
  }

  @Override
  public Stream<Object> getLabelSpace() {
    if (label == null) {
      return Stream.empty();
    }
    return Stream.of(label);
  }

  /**
   * Gets features.
   *
   * @return the features
   */
  public List<Feature> getFeatures() {
    return features;
  }

  /**
   * To vector vector.
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
    return vector;
  }

  @Override
  public Instance intern(Interner<String> interner) {
    return Instance.create(
      features.stream().map(f -> Feature.real(interner.intern(f.getName()), f.getValue())).collect(Collectors.toList()),
      label
    );
  }

  @Override
  public List<Instance> asInstances() {
    return Collections.singletonList(this);
  }


  @Override
  public void write(@NonNull StructuredWriter writer) throws IOException {
    boolean inArray = writer.inArray();
    if (inArray) writer.beginObject();
    writer.writeKeyValue("label", label);
    writer.beginObject("features");
    for (Feature f : features) {
      writer.writeKeyValue(f.getName(), f.getValue());
    }
    writer.endObject();
    if (inArray) writer.endObject();
  }

  @Override
  public void read(StructuredReader reader) throws IOException {
    this.label = null;
    this.features.clear();
    if (reader.peek() == ElementType.NAME) {
      this.label = reader.nextKeyValue("label");
    }
    reader.beginObject();
    while (reader.peek() != ElementType.END_OBJECT) {
      Tuple2<String, Val> fv = reader.nextKeyValue();
      this.features.add(Feature.real(fv.getKey(), fv.getValue().asDoubleValue()));
    }
    reader.endObject();
    this.features.trimToSize();
  }

}//END OF Instance
