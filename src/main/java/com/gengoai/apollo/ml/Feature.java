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

package com.gengoai.apollo.ml;

import com.gengoai.Copyable;
import com.gengoai.json.JsonEntry;
import com.gengoai.json.JsonSerializable;
import com.gengoai.string.Strings;
import lombok.NonNull;

import java.io.Serializable;
import java.util.Objects;

/**
 * <p>A feature is made up of a name and double value.</p> <p>Convention for binary predicates is
 * <code>PREFIX=PREDICATE</code> and when position is important <code>PREFIX[POSITION]=PREDICATE</code>.</p>
 *
 * @author David B. Bracewell
 */
public class Feature implements Serializable, Comparable<Feature>, Copyable<Feature>, JsonSerializable {
   private static final long serialVersionUID = 1L;
   private String name;
   private double value;

   private Feature(String name, double value) {
      this.name = name;
      this.value = value;
   }

   /**
    * Creates a binary feature with the value of TRUE (1.0)
    *
    * @param name the feature name
    * @return the feature
    */
   public static Feature TRUE(@NonNull String name) {
      return new Feature(name, 1.0);
   }

   /**
    * Creates a binary feature made up a prefix and one or more components with the value of TRUE (1.0). Feature name
    * will be in the form of <code>PREFIX=COMPONENT[1]_COMPONENT[2]_..._COMPONENT[N]</code>
    *
    * @param featurePrefix     the feature prefix
    * @param featureComponents the feature components
    * @return the feature
    */
   public static Feature TRUE(@NonNull String featurePrefix, @NonNull String... featureComponents) {
      return new Feature(featurePrefix + "=" + Strings.join(featureComponents, "_"), 1.0);
   }

   /**
    * Is false boolean.
    *
    * @param value the value
    * @return the boolean
    */
   public static boolean isFalse(double value) {
      return value == 0 || value == -1;
   }

   /**
    * Is false boolean.
    *
    * @param value the value
    * @return the boolean
    */
   public static boolean isFalse(String value) {
      return value.toLowerCase().equals("false");
   }

   /**
    * Is true boolean.
    *
    * @param value the value
    * @return the boolean
    */
   public static boolean isTrue(String value) {
      return value.toLowerCase().equals("true");
   }

   /**
    * Is true boolean.
    *
    * @param value the value
    * @return the boolean
    */
   public static boolean isTrue(double value) {
      return value == 1;
   }

   /**
    * Creates a real valued feature with the given name and value.
    *
    * @param name  the feature name
    * @param value the feature value
    * @return the feature
    */
   public static Feature real(@NonNull String name, double value) {
      return new Feature(name, value);
   }

   @Override
   public int compareTo(@NonNull Feature o) {
      return this.name.compareTo(o.name);
   }

   @Override
   public Feature copy() {
      return new Feature(name, value);
   }

   /**
    * Gets feature name.
    *
    * @return the feature name
    */
   public String getFeatureName() {
      return name;
   }

   /**
    * Gets name.
    *
    * @return the name
    */
   public String getName() {
      return this.name;
   }

   /**
    * Gets the predicate part of the feature.
    *
    * @return the predicate or the full name if no predicate is found
    */
   public String getPredicate() {
      int index = name.indexOf('=');
      if (index > 0) {
         return name.substring(index + 1);
      }
      return name;
   }

   /**
    * Gets the feature prefix.
    *
    * @return the prefix or an empty string if no prefix is specified
    */
   public String getPrefix() {
      int eqIndex = name.indexOf('=');
      int brIndex = name.indexOf('[');

      if (eqIndex <= 0 && brIndex <= 0) {
         return Strings.EMPTY;
      } else if (eqIndex <= 0) {
         return removePosition(name.substring(0, brIndex));
      } else if (brIndex <= 0) {
         return removePosition(name.substring(0, eqIndex));
      } else {
         return removePosition(name.substring(0, Math.min(eqIndex, brIndex)));
      }
   }

   /**
    * Gets value.
    *
    * @return the value
    */
   public double getValue() {
      return this.value;
   }


   @Override
   public boolean equals(Object o) {
      if (this == o) return true;
      if (!(o instanceof Feature)) return false;
      Feature feature = (Feature) o;
      return Double.compare(feature.value, value) == 0 &&
                Objects.equals(name, feature.name);
   }

   @Override
   public int hashCode() {

      return Objects.hash(name, value);
   }

   @Override
   public JsonEntry toJson() {
      return JsonEntry.object()
                      .addProperty("name", name)
                      .addProperty("value", value);
   }

   public static Feature fromJson(JsonEntry element) {
      return Feature.real(element.getStringProperty("name"),
                          element.getDoubleProperty("value"));
   }

   private String removePosition(String n) {
      return n.replaceAll("\\[[^\\]]+?\\]$", "");
   }

   @Override
   public String toString() {
      return "Feature(name=" + this.getName() + ", value=" + this.getValue() + ")";
   }
}//END OF Feature