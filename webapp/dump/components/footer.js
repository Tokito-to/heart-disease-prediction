class CustomFooter extends HTMLElement {
  connectedCallback() {
    this.attachShadow({ mode: 'open' });
    this.shadowRoot.innerHTML = `
      <style>
        footer {
          background: #1a202c;
          color: white;
          padding: 3rem 2rem;
        }
        .footer-content {
          max-width: 1200px;
          margin: 0 auto;
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
          gap: 2rem;
        }
        .footer-logo {
          font-size: 1.5rem;
          font-weight: bold;
          color: white;
          margin-bottom: 1rem;
          display: flex;
          align-items: center;
        }
        .footer-logo i {
          margin-right: 0.5rem;
        }
        .footer-description {
          color: #a0aec0;
          margin-bottom: 1.5rem;
        }
        .footer-links h3 {
          font-size: 1.125rem;
          font-weight: 600;
          margin-bottom: 1.25rem;
        }
        .footer-links ul {
          list-style: none;
          padding: 0;
          margin: 0;
        }
        .footer-links li {
          margin-bottom: 0.5rem;
        }
        .footer-links a {
          color: #a0aec0;
          text-decoration: none;
          transition: color 0.2s;
          display: flex;
          align-items: center;
        }
        .footer-links a:hover {
          color: #dc2626;
        }
        .footer-links a i {
          margin-right: 0.5rem;
          width: 16px;
        }
        .footer-bottom {
          text-align: center;
          padding-top: 2rem;
          margin-top: 2rem;
          border-top: 1px solid #2d3748;
          color: #a0aec0;
          font-size: 0.875rem;
        }
        .social-links {
          display: flex;
          gap: 1rem;
          margin-top: 1rem;
        }
        .social-links a {
          color: white;
          background: #2d3748;
          width: 36px;
          height: 36px;
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
          transition: background 0.2s;
        }
        .social-links a:hover {
          background: #dc2626;
        }
        @media (max-width: 768px) {
          .footer-content {
            grid-template-columns: 1fr;
          }
        }
      </style>
      <footer>
        <div class="footer-content">
          <div>
            <div class="footer-logo">
              <i data-feather="heart"></i>
              HeartGuard AI
            </div>
            <p class="footer-description">
              Using advanced artificial neural networks to predict heart disease risk and promote cardiovascular health awareness.
            </p>
            <div class="social-links">
              <a href="#"><i data-feather="twitter"></i></a>
              <a href="#"><i data-feather="facebook"></i></a>
              <a href="#"><i data-feather="instagram"></i></a>
              <a href="#"><i data-feather="linkedin"></i></a>
            </div>
          </div>
          <div class="footer-links">
            <h3>Quick Links</h3>
            <ul>
              <li><a href="#"><i data-feather="chevron-right"></i> Home</a></li>
              <li><a href="assessment.html"><i data-feather="chevron-right"></i> Risk Assessment</a></li>
              <li><a href="#disclaimer"><i data-feather="chevron-right"></i> Disclaimer</a></li>
              <li><a href="#team"><i data-feather="chevron-right"></i> Our Team</a></li>
</ul>
          </div>
          <div class="footer-links">
            <h3>Resources</h3>
            <ul>
              <li><a href="#"><i data-feather="chevron-right"></i> Heart Health Tips</a></li>
              <li><a href="#"><i data-feather="chevron-right"></i> Research Papers</a></li>
              <li><a href="#"><i data-feather="chevron-right"></i> FAQs</a></li>
              <li><a href="#"><i data-feather="chevron-right"></i> Contact Us</a></li>
            </ul>
          </div>
          <div class="footer-links">
            <h3>Legal</h3>
            <ul>
              <li><a href="#"><i data-feather="chevron-right"></i> Privacy Policy</a></li>
              <li><a href="#"><i data-feather="chevron-right"></i> Terms of Service</a></li>
              <li><a href="#"><i data-feather="chevron-right"></i> Data Usage</a></li>
              <li><a href="#"><i data-feather="chevron-right"></i> Cookie Policy</a></li>
            </ul>
          </div>
        </div>
        <div class="footer-bottom">
          <p>&copy; 2023 HeartGuard AI. All rights reserved. This tool is for informational purposes only and not medical advice.</p>
        </div>
      </footer>
    `;
  }
}
customElements.define('custom-footer', CustomFooter);

