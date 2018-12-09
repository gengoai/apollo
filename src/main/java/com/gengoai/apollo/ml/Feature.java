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

import java.io.Serializable;
import java.util.Objects;

/**
 * <p>A feature is made up of a name and double value. By convention, features should have a name with a prefix given
 * in the following manner <code>PREFIX=NAME</code>. The prefix is used for generating contextual features in which the
 * features from the previous or next examples are needed.</p>
 *
 * @author David B. Bracewell
 */
public class Feature implements Serializable, Comparable<Feature>, Copyable<Feature> {
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
      this.name = name;
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
    * Creates a real valued feature with the given name and value.
    *
    * @param name  the feature name
    * @param value the feature value
    * @return the feature
    */
   public static Feature realFeature(String name, double value) {
      return new Feature(name, value);
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

   @Override
   public int hashCode() {
      return Objects.hash(name, value);
   }

   /**
    * Checks if the name of this feature starts with the given prefix (case-sensitive)
    *
    * @param prefix the prefix to check
    * @return True if this feature starts with the given prefix, False otherwise
    */
   public boolean hasPrefix(String prefix) {
      return name.startsWith(prefix);
   }


   @Override
   public String toString() {
      return "Feature(name=" + name + ", value=" + value + ")";
   }
}//END OF Feature