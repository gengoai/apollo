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

package com.davidbracewell.apollo.similarity;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.linalg.VectorMap;
import com.davidbracewell.collection.Counter;
import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import com.google.common.collect.Maps;
import lombok.NonNull;

import java.io.Serializable;
import java.util.Map;
import java.util.Set;

/**
 * The type Distance measure.
 *
 * @author David B. Bracewell
 */
public abstract class DistanceMeasure implements Serializable {

  private static final long serialVersionUID = 1L;

  /**
   * Calculate double.
   *
   * @param v1 the v 1
   * @param v2 the v 2
   * @return the double
   */
  public final double calculate(@NonNull Counter<?> v1, @NonNull Counter<?> v2) {
    return calculate(v1.asMap(), v2.asMap());
  }

  /**
   * Calculate double.
   *
   * @param s1 the s 1
   * @param s2 the s 2
   * @return the double
   */
  public final double calculate(@NonNull Set<?> s1, @NonNull Set<?> s2) {
    return calculate(
      Maps.asMap(s1, SET_NUMBER.INSTANCE),
      Maps.asMap(s2, SET_NUMBER.INSTANCE)
    );
  }

  /**
   * Calculate double.
   *
   * @param v1 the v 1
   * @param v2 the v 2
   * @return the double
   */
  public final double calculate(@NonNull Vector v1, @NonNull Vector v2) {
    Preconditions.checkArgument(v1.dimension() == v2.dimension(), "Dimension mismatch.");
    return calculate(VectorMap.wrap(v1), VectorMap.wrap(v2));
  }

  /**
   * Calculate double.
   *
   * @param m1 the m 1
   * @param m2 the m 2
   * @return the double
   */
  public abstract double calculate(Map<?, ? extends Number> m1, Map<?, ? extends Number> m2);

  /**
   * Calculate double.
   *
   * @param v1 the v 1
   * @param v2 the v 2
   * @return the double
   */
  public abstract double calculate(double[] v1, double[] v2);

  private enum SET_NUMBER implements Function<Object, Number> {
    INSTANCE;

    @Override
    public Number apply(Object input) {
      return 1d;
    }

  }

}//END OF DistanceMeasure
