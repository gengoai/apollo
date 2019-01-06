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

import com.gengoai.Copyable;
import com.gengoai.Interner;
import com.gengoai.string.Strings;

import java.io.Serializable;
import java.util.Objects;

/**
 * <p>
 * A feature is made up of a name and double value. By convention, features should have a name with a prefix given in
 * the following manner <code>PREFIX=SUFFIX</code>, where the suffix is the name/value. A more concrete example is
 * <code>WORD=the</code> and <code>POS=NP</code>, where the prefixes are <code>WORD</code> and <code>POS</code> and the
 * suffixes (or names) are <code>the</code> and <code>NP</code> respectively.
 * </p>
 * <p>
 * This convention of <code>PREFIX=SUFFIX</code> is used for generating contextual features using a {@link
 * ContextFeaturizer}s in which the features with a given prefix from the previous or next examples are needed.
 * </p>
 * <p>
 * All feature names (<code>PREFIX=SUFFIX</code>) are interned to reduce the memory footprint of millions of features
 * being created. Note: that the interning is not distributed and will be duplicated on each node in the cluster.
 * </p>
 *
 * @author David B. Bracewell
 */
public final class Feature implements Serializable, Comparable<Feature>, Copyable<Feature> {
   private static final Interner<String> featureInterner = new Interner<>();
   private static final long serialVersionUID = 1L;
   /**
    * The name of the feature (e.g. <code>W=Apollo</code>)
    */
   public final String name;
   /**
    * The value of the feature
    */
   public final double value;

   private Feature(String name, double value) {
      this.name = featureInterner.intern(name);
      this.value = value;
   }

   /**
    * Creates a boolean valued feature with the given name and value of <code>1.0</code>.
    *
    * @param name the feature name
    * @return the feature
    */
   public static Feature booleanFeature(String name) {
      return new Feature(name, 1.0);
   }

   /**
    * Creates a boolean valued feature with the given name and value of <code>1.0</code>.
    *
    * @param prefix the feature prefix
    * @param name   the feature name
    * @return the feature
    */
   public static Feature booleanFeature(String prefix, String name) {
      return new Feature(prefix + "=" + name, 1.0);
   }

   /**
    * Gets the feature prefix, which is the string from the start to the index of the first equal sign.
    *
    * @param name the full feature name
    * @return the prefix if one, otherwise the full feature name.
    */
   public static String getPrefix(String name) {
      int eqIndex = name.indexOf('=');
      if (eqIndex > 0) {
         return name.substring(0, eqIndex);
      }
      return name;
   }

   /**
    * Gets the feature suffix, which is the string from the end of the first equal sign to the end of the string.
    *
    * @param name the full feature name
    * @return the suffix if one, otherwise the full feature name.
    */
   public static String getSuffix(String name) {
      int eqIndex = name.indexOf('=');
      if (eqIndex > 0) {
         return name.substring(eqIndex + 1);
      }
      return name;
   }

   /**
    * Checks if the given value is false (i.e. is not true)
    *
    * @param value the value
    * @return True if the string represents a false value, False otherwise
    */
   public static boolean isFalse(String value) {
      return !isTrue(value);
   }

   /**
    * Checks if the given value is false (i.e. is not true)
    *
    * @param value the value
    * @return True if the double is not 1;
    */
   public static boolean isFalse(double value) {
      return !isTrue(value);
   }

   /**
    * Checks if the given value is true
    *
    * @param value the value
    * @return True if the double is 1;
    */
   public static boolean isTrue(double value) {
      return value == 1;
   }

   /**
    * Checks if the given value is true (case-insensitive match to "true")
    *
    * @param value the value
    * @return True if the string represents a true value, False otherwise
    */
   public static boolean isTrue(String value) {
      return value.toLowerCase().equals("true");
   }

   /**
    * Creates a real valued feature with the given name and value.
    *
    * @param name  the feature name
    * @param value the feature value
    * @return the feature
    */
   public static Feature realFeature(String name, double value) {
      return new Feature(name, value);
   }

   /**
    * Creates a real valued feature with the given name and value.
    *
    * @param prefix the feature prefix
    * @param name   the feature name
    * @param value  the feature value
    * @return the feature
    */
   public static Feature realFeature(String prefix, String name, double value) {
      return new Feature(prefix + "=" + name, value);
   }

   @Override
   public int compareTo(Feature o) {
      return this.name.compareTo(o.name);
   }

   @Override
   public Feature copy() {
      return new Feature(name, value);
   }

   @Override
   public boolean equals(Object o) {
      if (this == o) return true;
      if (!(o instanceof Feature)) return false;
      Feature feature = (Feature) o;
      return Double.compare(feature.value, value) == 0 &&
                Objects.equals(name, feature.name);
   }

   /**
    * Gets the feature name.
    *
    * @return the name of the feature
    */
   public String getName() {
      return name;
   }

   /**
    * Gets the feature prefix, which is the string from the start to the index of the first equal sign.
    *
    * @return the prefix if one, otherwise the full feature name.
    */
   public String getPrefix() {
      return getPrefix(name);
   }

   /**
    * Gets the feature suffix, which is the string from the end of the first equal sign to the end of the string.
    *
    * @return the suffix if one, otherwise the full feature name.
    */
   public String getSuffix() {
      return getSuffix(name);
   }

   /**
    * Gets the value of the feature.
    *
    * @return the value
    */
   public double getValue() {
      return value;
   }

   /**
    * Checks if the name of this feature starts with the given prefix (case-sensitive)
    *
    * @param prefix the prefix to check
    * @return True if this feature starts with the given prefix, False otherwise
    */
   public boolean hasPrefix(String prefix) {
      prefix = Strings.appendIfNotPresent(prefix, "=");
      return name.startsWith(prefix) || (name + "=").equals(prefix);
   }

   @Override
   public int hashCode() {
      return Objects.hash(name);
   }

   @Override
   public String toString() {
      return "Feature(" + name + " => " + value + ")";
   }


}//END OF Feature