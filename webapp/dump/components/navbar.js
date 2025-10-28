class CustomNavbar extends HTMLElement {
  connectedCallback() {
    this.attachShadow({ mode: 'open' });
    this.shadowRoot.innerHTML = `
      <style>
        nav {
          background: white;
          padding: 1rem 2rem;
          display: flex;
          justify-content: space-between;
          align-items: center;
          box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
          position: sticky;
          top: 0;
          z-index: 1000;
        }
        .logo {
          color: #dc2626;
          font-weight: bold;
          font-size: 1.5rem;
          display: flex;
          align-items: center;
        }
        .logo i {
          margin-right: 0.5rem;
        }
        ul {
          display: flex;
          gap: 1.5rem;
          list-style: none;
          margin: 0;
          padding: 0;
        }
        a {
          color: #4b5563;
          text-decoration: none;
          font-weight: 500;
          transition: color 0.2s;
          display: flex;
          align-items: center;
        }
        a:hover {
          color: #dc2626;
        }
        a i {
          margin-right: 0.3rem;
        }
        .cta {
          background-color: #dc2626;
          color: white !important;
          padding: 0.5rem 1.25rem;
          border-radius: 0.375rem;
          transition: background-color 0.2s;
        }
        .cta:hover {
          background-color: #b91c1c;
        }
        @media (max-width: 768px) {
          nav {
            flex-direction: column;
            padding: 1rem;
          }
          ul {
            margin-top: 1rem;
            flex-wrap: wrap;
            justify-content: center;
          }
        }
      </style>
      <nav>
        <a href="#" class="logo">
          <i data-feather="heart"></i>
          HeartGuard AI
        </a>
        <ul>
          <li><a href="#"><i data-feather="home"></i> Home</a></li>
          <li><a href="assessment.html"><i data-feather="activity"></i> Predict</a></li>
          <li><a href="#disclaimer"><i data-feather="alert-triangle"></i> Disclaimer</a></li>
          <li><a href="#team"><i data-feather="users"></i> Team</a></li>
          <li><a href="assessment.html" class="cta"><i data-feather="shield"></i> Get Started</a></li>
</ul>
      </nav>
    `;
  }
}
customElements.define('custom-navbar', CustomNavbar);

