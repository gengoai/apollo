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

package com.gengoai.apollo.ml.params;

import com.gengoai.Copyable;
import com.gengoai.conversion.Cast;
import com.gengoai.logging.Logger;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Comparator;

import static com.gengoai.collection.Arrays2.binarySearch;

/**
 * <p>Defines a set of parameters and values.</p>
 *
 * @author David B. Bracewell
 */
public class ParamMap implements Serializable, Copyable<ParamMap> {
   private static final Logger log = Logger.getLogger(ParamMap.class);
   private ParamValuePair[] map;

   /**
    * Instantiates a new Param map.
    *
    * @param pvps the parameter value pairs making up the map (parameters and default values)
    */
   public ParamMap(ParamValuePair... pvps) {
      this.map = pvps;
      Arrays.sort(this.map, Comparator.comparing(p -> p.param.name));
   }

   @Override
   public ParamMap copy() {
      return Copyable.deepCopy(this);
   }

   /**
    * Gets the value of the given parameter.
    *
    * @param <T>   the type parameter
    * @param param the parameter whose value we want
    * @return the value of the parameter
    * @throws IllegalArgumentException if the parameter is not contained in the map
    */
   public <T> T get(Param<T> param) {
      int idx = indexOf(param);
      if (idx < 0) {
         throw new IllegalArgumentException();
      }
      return Cast.as(map[idx].value);
   }

   public <T> T getOrDefault(Param<T> param, T defaultValue) {
      int idx = indexOf(param);
      if (idx < 0) {
         return defaultValue;
      }
      return Cast.as(map[idx].value);
   }

   private int indexOf(Param param) {
      return binarySearch(map, param,
                          (ParamValuePair<?> pvp, Param<?> p) -> pvp.param.name.compareTo(p.name));
   }


   public void merge(ParamMap other) {
      for (ParamValuePair paramValuePair : other.map) {
         set(paramValuePair);
      }
   }

   public ParamMap set(ParamValuePair paramValuePair) {
      int idx = indexOf(paramValuePair.param);
      if (idx < 0) {
         log.warn("Invalid Parameter: {0}" + paramValuePair.param.name);
         return this;
      }
      map[idx] = paramValuePair;
      return this;
   }

   /**
    * Updates the values in the parameter map.
    *
    * @param pvp the parameter value pairs used to update.
    */
   public void update(ParamValuePair... pvp) {
      for (ParamValuePair paramValuePair : pvp) {
         set(paramValuePair);
      }
   }


   /**
    * Sets the value for the given parameter name.
    *
    * @param name  the name
    * @param value the value
    */
   @SuppressWarnings("unchecked")
   public ParamMap set(String name, Object value) {
      int idx = binarySearch(map, name, (pvp, s) -> pvp.param.name.compareTo(s));
      if (idx < 0) {
         log.warn("Invalid Parameter: {0}" + name);
         return this;
      }
      map[idx] = map[idx].param.set(value);
      return this;
   }


}//END OF ParamMap
