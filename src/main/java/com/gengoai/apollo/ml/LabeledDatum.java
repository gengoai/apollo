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

import lombok.NonNull;

import java.io.Serializable;

/**
 * Simple container for an object and its associated label
 *
 * @param <T> the type parameter
 * @author David B. Bracewell
 */
public class LabeledDatum<T> implements Serializable {
   private static final long serialVersionUID = 1L;
   private Object label;
   private T data;

   private LabeledDatum(Object label, T data) {
      this.label = label;
      this.data = data;
   }

   /**
    * Convenience method for creating a labeled data point
    *
    * @param <R>   the data type parameter
    * @param label the label
    * @param data  the data
    * @return the labeled data
    */
   public static <R> LabeledDatum<R> of(Object label, @NonNull R data) {
      return new LabeledDatum<>(label, data);
   }

   public boolean equals(Object o) {
      if (o == this) return true;
      if (!(o instanceof LabeledDatum)) return false;
      final LabeledDatum other = (LabeledDatum) o;
      final Object this$label = this.getLabel();
      final Object other$label = other.getLabel();
      if (this$label == null ? other$label != null : !this$label.equals(other$label)) return false;
      final Object this$data = this.getData();
      final Object other$data = other.getData();
      if (this$data == null ? other$data != null : !this$data.equals(other$data)) return false;
      return true;
   }

   /**
    * Gets data.
    *
    * @return the data
    */
   public T getData() {
      return this.data;
   }

   /**
    * Gets label.
    *
    * @return the label
    */
   public Object getLabel() {
      return this.label;
   }

   @Override
   public int hashCode() {
      final int PRIME = 59;
      int result = 1;
      final Object $label = this.getLabel();
      result = result * PRIME + ($label == null ? 43 : $label.hashCode());
      final Object $data = this.getData();
      result = result * PRIME + ($data == null ? 43 : $data.hashCode());
      return result;
   }

   @Override
   public String toString() {
      return "LabeledDatum(label=" + this.getLabel() + ", data=" + this.getData() + ")";
   }
}//END OF LabeledDatum
