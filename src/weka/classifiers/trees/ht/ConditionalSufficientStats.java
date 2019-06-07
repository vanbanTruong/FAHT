/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 *    ConditionalSufficientStats.java
 *    Copyright (C) 2013 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.trees.ht;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

/**
 * Records sufficient stats for an attribute
 * 
 * @author Richard Kirkby (rkirkby@cs.waikato.ac.nz)
 * @author Mark Hall (mhall{[at]}pentaho{[dot]}com)
 * @version $Revision$
 */
public abstract class ConditionalSufficientStats implements Serializable {

	/**
	 * For serialization
	 */
	private static final long serialVersionUID = 8724787722646808376L;

	// Discrimination calculation related
	//String deprivedAttName = "Female";
	//String undeprivedAttName = "Male";
	String grantedString = ">50k"; // // <=50K: 0, >50K: 1

	int deprivedIndex = 1;    //M:0 F:1
	int undeprivedIndex = 0;

	// Lookup by discrimination stats for numeric attribute
	String deprivedTotalDist = "deprivedTotalDist";
	String undeprivedTotalDist = "undeprivedTotalDist";
	String deprivedGrantedDist = "deprivedGrantedDist";
	String undeprivedGrantedDist = "undeprivedGrantedDist";

	/** Lookup by class value */
	protected Map<String, Object> m_classLookup = new HashMap<String, Object>();

	/** added: Lookup by attribute value for nominal attribute */
	protected Map<Integer, Object> nomlAttValLookup = new HashMap<Integer, Object>();

	/** added: discrimination related distribution for numeric attribute */
	protected Map<String, Object> numGaussDistLookup = new HashMap<String, Object>() {
		{
			put(deprivedTotalDist, null);
			put(undeprivedTotalDist, null);
			put(deprivedGrantedDist, null);
			put(undeprivedGrantedDist, null);
		}
	};

	/**
	 * Update this stat with the supplied attribute value and class value
	 * 
	 * @param attVal
	 *            the value of the attribute
	 * @param classVal
	 *            the class value
	 * @param weight
	 *            the weight of this observation
	 * @param senAttVal
	 *            added for the value of sensitive attribute
	 */
	public abstract void update(double attVal, String classVal, double weight, int senAttVal);

	/**
	 * Return the probability of an attribute value conditioned on a class value
	 * 
	 * @param attVal
	 *            the attribute value to compute the conditional probability for
	 * @param classVal
	 *            the class value
	 * @return the probability
	 */
	public abstract double probabilityOfAttValConditionedOnClass(double attVal, String classVal);

	/**
	 * Return the best split
	 * 
	 * @param splitMetric
	 *            the split metric to use
	 * @param preSplitDist
	 *            the distribution of class values prior to splitting
	 * @param attName
	 *            the name of the attribute being considered for splitting
	 * @return the best split for the attribute
	 */
	public abstract SplitCandidate bestSplit(SplitMetric splitMetric, Map<String, WeightMass> preSplitDist,
			String attName, ActiveHNode node);
}
